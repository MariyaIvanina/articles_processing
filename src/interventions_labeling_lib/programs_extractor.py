import spacy
from text_processing import text_normalizer
from utilities import excel_writer
from utilities import excel_reader
import re
import os
import nltk

nlp = spacy.load('en_core_web_sm')

class ProgramExtractor:

    def __init__(self, filter_word_list):
        self.words_for_programs = ["program", "programme", "initiative", "project"]
        self.filter_word_list = filter_word_list

    def is_filter_word_main(self, word_expression):
        doc = nlp(word_expression)
        for chunk in doc.noun_chunks:
            if chunk.root.text in self.words_for_programs or text_normalizer.is_abbreviation(chunk.root.text):
                return True
        return False

    def list_2_str(self, abbr_words):
        return ";".join([ab_w[0] + " # " + ab_w[1] for ab_w in abbr_words])

    def delete_all_words_after_program_words(self, word):
        new_words = []
        program_word_is_found = False
        for w in word.split():
            if not program_word_is_found or text_normalizer.is_abbreviation(w):
                new_words.append(w)
            if w in self.words_for_programs:
                program_word_is_found= True 
        return " ".join(new_words) if len(new_words) > 1 else ""

    def get_noun_chunk(self, word_expression):
        doc = nlp(word_expression)
        has_abb_word = False
        for w in word_expression.split():
            if text_normalizer.is_abbreviation(w):
                has_abb_word = True
        if has_abb_word:
            return self.delete_all_words_after_program_words(word_expression)
        for chunk in doc.noun_chunks:
            if chunk.root.text in self.words_for_programs:
                return chunk.text if len(chunk.text.split()) > 1 else self.delete_all_words_after_program_words(word_expression)
        return ""

    def clean_from_too_frequent_words(self, word_expression):
        words_to_join = []
        all_to_join = False 
        for w in word_expression.split():
            if len(w) == 1:
                continue
            if w in text_normalizer.stopwords_all or w in ["ass"]:
                continue
            if all_to_join or text_normalizer.get_rank_of_word(w) >= 0.6 or w in self.words_for_programs:
                words_to_join.append(w)
                all_to_join = True
        return " ".join(words_to_join) if len(words_to_join) > 1 else ""

    def clean_abbr_word(self, abbr_words):
        filtered_abbr_words = []
        for abbr_word in abbr_words:
            for program_word in self.words_for_programs:
                if re.search(r"\b%s\b"%program_word, abbr_word[1]) != None:
                    filtered_abbr_words.append(abbr_word)
        return filtered_abbr_words

    def find_programs(self, hyponyms_search, _abbreviations_resolver, file_with_found_programs="../data/extracted_programs.xlsx",
            file_with_found_programs_filtered="../data/extracted_programs_filtered.xlsx"):
        programs_raw_extracted = set()
        for key in hyponyms_search.dict_hyponyms.keys():
            for program_word in self.words_for_programs:
                for key_word in hyponyms_search.dict_hyponyms[key]:
                    for key_w in key.split(";"):
                        if re.search(r"\b%s\b"%program_word, key_w) != None and key_w not in self.words_for_programs:
                            programs_raw_extracted.add(key_w)
                    for key_w in key_word.split(";"):
                        if re.search(r"\b%s\b"%program_word,key_w) != None and key_w not in self.words_for_programs:
                            programs_raw_extracted.add(key_w)
        programs_filtered = set()
        programs_found = set()
        if os.path.exists(file_with_found_programs):
            programs_found = set(excel_reader.ExcelReader().read_df_from_excel(file_with_found_programs)["Programme"].values)
        if os.path.exists(file_with_found_programs_filtered):
            programs_filtered = set(excel_reader.ExcelReader().read_df_from_excel(file_with_found_programs_filtered)["Programme"].values)
        for program in programs_raw_extracted:
            if self.is_filter_word_main(program):
                abbr_word = self.list_2_str(self.clean_abbr_word(_abbreviations_resolver.find_abbreviations_meanings_in_text(program)))
                res = abbr_word if abbr_word != "" else self.clean_from_too_frequent_words(self.get_noun_chunk(program))
                if res != "":
                    programs_found.add(res)
                else:
                    programs_filtered.add(program)
        excel_writer.ExcelWriter().save_data_in_excel(list(programs_found), ["Programme"], file_with_found_programs)
        excel_writer.ExcelWriter().save_data_in_excel(list(programs_filtered), ["Programme"], file_with_found_programs_filtered)

    def label_articles_by_keywords_search(self, articles_df, search_engine_inverted_index, _abbreviations_resolver,
            program_filename = "../data/extracted_programs.xlsx", column_name="programs_found"):
        programs_map = excel_reader.ExcelReader().read_df_from_excel(program_filename)["Programme"].values
        articles_df[column_name] = ""
        for program in programs_map:
            program = re.sub(r"\bprogramme\b", "program", program)
            program_name, abbreviation_meanings = (program.split("#")[0].strip(), program.split("#")[1].strip()) if len(program.split("#")) > 1 else (program, "")
            for article_index in search_engine_inverted_index.find_articles_with_keywords(
                    [program_name], 0.92, extend_with_abbreviations = True, extend_abbr_meanings = abbreviation_meanings):
                if articles_df[column_name].values[article_index] == "":
                    articles_df[column_name].values[article_index] = set()
                articles_df[column_name].values[article_index].add(_abbreviations_resolver.replace_abbreviations(program_name, abbreviation_meanings))
        articles_df = text_normalizer.replace_string_default_values(articles_df, column_name)
        return articles_df

    def label_articles_with_programs(self, articles_df, search_engine_inverted_index, _abbreviations_resolver,
            program_filename = "../data/extracted_programs.xlsx",
            model_type="model",
            model_folder="../tmp/programs_extraction_model_2619",
            column_name="programs_found",
            columns_to_process=["title", "abstract"]):
        if model_type == "keywords":
            return self.label_articles_by_keywords_search(articles_df, search_engine_inverted_index, _abbreviations_resolver,
            program_filename=program_filename, column_name=column_name)
        elif model_type == "model":
            program_ner_model = spacy.load(model_folder)
            programs_to_filter = ["ACRONYMS", "FROM","LIST","SUMMARY", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS", "APPENDIX", "APPENDICES",
                     "ACHIEVEMENTS", "TOTAL", "PART", "INCONSISTENCIES", "INTERVENTIONS", "GOOD", "TASKS",
                     "FINDINGS", "UNITED", "DONORS", "EXPECTED", "REQUESTS", "LABORATORY", "LEARNED", "KNOW", "AFGHANISTAN",
                     "SPECIFICALLY", "DATE", "INDIA", "KNOW", "CANNOT", "MONTHLY", "CLOSEST", "PROJECT", "PROGRAM",
                     "TARGETS", "INDICATORS", "USED", "DESIGN"]
            articles_df[column_name] = ""
            for i in range(len(articles_df)):
                programs_found = []
                for column in columns_to_process:
                    for sentence in nltk.sent_tokenize(articles_df[column].values[i]):
                        for res in program_ner_model(sentence).ents:
                            programs_found.append(res.text.strip())
                programs_found = list(set(programs_found))
                programs = []
                for program in programs_found:
                    program = re.sub(r"\bprogramme\b", "program", program)
                    program = _abbreviations_resolver.replace_abbreviations(text=text_normalizer.normalize_sentence(program))
                    if program.strip() and program not in programs_to_filter and len(program) > 1 and len(program.split()) < 15:
                        programs.append(program.strip())
                articles_df[column_name].values[i] = programs
        else:
            print("Model type can be either 'model' or 'keywords', but was ", model_type)
        return articles_df


