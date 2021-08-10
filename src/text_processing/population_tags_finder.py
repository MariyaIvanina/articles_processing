from text_processing import text_normalizer

class PopulationTagsFinder:

    def __init__(self, columns_to_process=["title","abstract","keywords","identificators"]):
        self.small_scale_farmers_dictionary = ["smallhold", "smallholder","small farm","small farmer", "microfarm", "pastoral", \
                                          "pastoralist", "family run farm", "familyrun farm","family owned farm","familyowned farm", \
                                         "family managed farm","familymanaged farm", "agropastoral", "agropastoralist", "ejido",\
                                         "small holding", "small land holding",  "small landholding"]
        self.small_scale_keywords = ["small scale", "smallscale", "low income", "lowincome", "subsistence", "resource poor", "resourcepoor",\
                               "resource limited", "small size", "smallsize", "peasant"]
        self.farm_markers = ["farm", "farmer", "mixed farm", "mixed farmer", "mixedfarm", "mixedfarmer", "agriculture", "producer", "produce",\
                       "grower", "agronomy", "husbandry", "aquaculture", "floriculture", "horticulture"]
        self.columns_to_process = columns_to_process

    def find_positions_of_words(self, text_words, marker_words_set):
        positions_found = set()
        for i in range(len(text_words)):
            for offset in range(3):
                text_to_check = " ".join(text_words[i:i+offset])
                if text_to_check in marker_words_set:
                    for pos_offset in range(offset):
                        positions_found.add(i+pos_offset)
        return list(positions_found)

    def has_neigbours_with_offset(self, first_positions, second_positions,neighbors):
        for first_pos in first_positions:
            for second_pos in second_positions:
                if abs(first_pos - second_pos) <= neighbors:
                    return True
        return False

    def find_all_markers(self, marker_words, search_engine_inverted_index, threshold=0.9):
        marker_words_set = set()
        for word in marker_words:
            marker_words_set = marker_words_set.union(search_engine_inverted_index.find_similar_words_by_spelling(word, threshold))
        return marker_words_set

    def filter_with_nearest_words(self, docs_to_check, articles_df, farm_markers, small_scale_keywords, search_engine_inverted_index, neighbors = 5):
        filtered_docs = set()
        small_scale_keywords_markers = self.find_all_markers(small_scale_keywords, search_engine_inverted_index, threshold=0.9)
        farmer_markers_markers = self.find_all_markers(farm_markers, search_engine_inverted_index, threshold=0.9)
        for article in docs_to_check:
            text = ""
            for column in self.columns_to_process:
                if column in ["keywords","identificators"]:
                    text = text + " . " + (text_normalizer.normalize_key_words_for_search(articles_df[column].values[article]) if column in articles_df.columns else "" )
                else:
                    text = text + " . " + text_normalizer.normalize_text(articles_df[column].values[article])
            text_words = text_normalizer.get_stemmed_words_inverted_index(text)
            small_scale_positions = self.find_positions_of_words(text_words, small_scale_keywords_markers)
            farmer_markers_positions = self.find_positions_of_words(text_words, farmer_markers_markers)
            if self.has_neigbours_with_offset(small_scale_positions, farmer_markers_positions, neighbors):
                filtered_docs.add(article)
        return filtered_docs

    def label_with_population_tags(self, articles_df, search_engine_inverted_index):
        articles_df["population tags"] = ""
        for i in range(len(articles_df)):
            articles_df["population tags"].values[i] = set()
        articles_with_smallholders = set()

        for key_word in self.small_scale_farmers_dictionary:
            for article in search_engine_inverted_index.find_articles_with_keywords([key_word],0.9, extend_with_abbreviations = False):
                articles_df["population tags"].values[article].add("Small scale farmers")
                articles_with_smallholders.add(article)

        docs_with_small_scale = set()
        for key_word in self.small_scale_keywords:
            for article in search_engine_inverted_index.find_articles_with_keywords([key_word],0.9, extend_with_abbreviations = False):
                docs_with_small_scale.add(article)


        docs_with_small_scale = docs_with_small_scale - articles_with_smallholders
        docs_with_small_scale = self.filter_with_nearest_words(docs_with_small_scale, articles_df, self.farm_markers, self.small_scale_keywords, search_engine_inverted_index)
        for article in docs_with_small_scale:
            articles_df["population tags"].values[article].add("Small scale farmers")
            articles_with_smallholders.add(article)


        articles_with_farmers = set(search_engine_inverted_index.find_articles_with_keywords(["farmer"],0.9, extend_with_abbreviations = False))
        for article in (articles_with_farmers - articles_with_smallholders):
            articles_df["population tags"].values[article].add("Farmers")
        articles_df = text_normalizer.replace_string_default_values(articles_df, "population tags")

        return articles_df