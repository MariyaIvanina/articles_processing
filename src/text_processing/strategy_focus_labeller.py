from utilities import excel_reader
from text_processing import text_normalizer

class StrategyFocusLabeller:

    def __init__(self, dict_filename, status_logger = None):
        self.status_logger = status_logger
        self.df_strategic_words = excel_reader.ExcelReader().read_file(dict_filename)

    def log_percents(self, percent):
        if self.status_logger is not None:
            self.status_logger.update_step_percent(percent)

    def label_articles_with_strategies(self, label_column, articles_df, 
            search_engine_inverted_index, _abbreviations_resolver, label_details_column = "", 
            threshold=0.92, print_all = True):
        articles_df[label_column] = ""
        articles_df[label_details_column] = ""
        for i in range(len(articles_df)):
            articles_df[label_column].values[i] = set()
            articles_df[label_details_column].values[i] = set()
        for i in range(len(self.df_strategic_words)):
            labels = [w.strip() for w in self.df_strategic_words["Intervention label in sentence"].values[i].split(";")]
            high_label = self.df_strategic_words["High level label"].values[i].strip()
            docs_found = set()
            first_assign = True
            for key_word in self.df_strategic_words["Keyword"].values[i].split(";"):
                if first_assign:
                    docs_found = search_engine_inverted_index.find_articles_with_keywords([key_word])
                    first_assign = False
                else:
                    docs_found = docs_found.intersection(search_engine_inverted_index.find_articles_with_keywords([key_word],
                        threshold=threshold))
            for ind in docs_found:
                if len(set(articles_df["intervention_labels"].values[ind]).intersection(labels)):
                    articles_df[label_column].values[ind].add(high_label)
                    articles_df[label_details_column].values[ind].add(self.df_strategic_words["Keyword"].values[i])
        articles_df = text_normalizer.replace_string_default_values(articles_df, label_column)
        if label_details_column != "":
            articles_df = text_normalizer.replace_string_default_values(articles_df, label_details_column)
        sentence_dict = {}
        for i in range(len(articles_df)):
            if articles_df["sentence"].values[i] not in sentence_dict:
                sentence_dict[articles_df["sentence"].values[i]] = set()
            sentence_dict[articles_df["sentence"].values[i]].update(articles_df["intervention_labels"].values[i])
        dict_by_label = {
            ("Sanitation/Hygiene","Community and behavior"): ["2.c Hygiene Social and Behavior Change"],
            ("Water infrastructure", "Community and behavior"): ["3.a Professionalization of Rural Services"],
            ("Sustainability/Environmental health", "Community and behavior"): ["4.c Water use efficiency & conservation"],
            ("Sanitation/Hygiene","Assessment tool or program"): ["2.a Sanitation Demand Creation"],
            ("Water quality", "Assessment tool or program"): ["3.e Drinking water quality (WQ)"],
            ("Water infrastructure", "Assessment tool or program"): [
                ("3.b Urban water utilities", ['urbanization', 'urbanized', 'suburban', 'peri urban', 'city', 'urbanism', 'urban']),
                ("3.c Rural Water Monitoring", ['peasant', 'village', 'small town', 'countryside', 'rural'])],
            ("Sustainability/Environmental health", "Assessment tool or program"): ["4.f Water resources regulation"],
            ("Menstrual hygiene management"): ["2.e MHM"],
            ("Fecal sludge management"): ["2.d Fecal Sludge Management"],
            ("Fecal sludge management", "Community and behavior"): ["2.c Hygiene Social and Behavior Change"],
            ("Water quality", "Community and behavior"): ["2.c Hygiene Social and Behavior Change"]
        }

        keywords_sentences_ids = {}

        for i in range(len(articles_df)):
            all_sent_labels = sentence_dict[articles_df["sentence"].values[i]]
            for _l in dict_by_label:
                for label_info in dict_by_label[_l]:
                    if type(label_info) == str:
                        label_, keywords = label_info, []
                    else:
                        label_, keywords = label_info
                    if len(keywords):
                        keywords_joined = ";".join(keywords)
                        if keywords_joined not in keywords_sentences_ids:
                            keywords_sentences_ids[keywords_joined] = set()
                            for key_word in keywords:
                                keywords_sentences_ids[keywords_joined] = keywords_sentences_ids[keywords_joined].union(
                                    search_engine_inverted_index.find_articles_with_keywords([key_word], threshold=threshold))
                    if len(set(articles_df["intervention_labels"].values[i]).intersection(set(_l))):
                        if len(all_sent_labels.intersection(set(_l))) == len(_l):
                            if len(keywords):
                                if i in keywords_sentences_ids[";".join(keywords)]:
                                    articles_df[label_column].values[i].append(label_)
                            else:
                                articles_df[label_column].values[i].append(label_)
        for i in range(len(articles_df)):
            articles_df[label_column].values[i] = list(set(articles_df[label_column].values[i]))
        total_labelled = 0
        if print_all:
            dict_found = {}
            for i in range(len(articles_df)):
                if len(articles_df[label_column].values[i]):
                    for _l in articles_df[label_column].values[i]:
                        if _l not in dict_found:
                            dict_found[_l] = 0
                        dict_found[_l] += 1
                    total_labelled += 1
            for gr in sorted(dict_found.items(), key=lambda x: x[1], reverse=True):
                print(gr[0], gr[1])
        print("Labelled docs with strategies: %d" % total_labelled)
        return articles_df

