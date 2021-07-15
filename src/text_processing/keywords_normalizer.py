from text_processing import text_normalizer

class KeywordsNormalizer:

    def __init__(self):
        pass

    def normalize_key_words(self, articles_df, new_column_name = "normalized_key_words", key_words_column_names = ["keywords", "identificators"]):
        articles_df[new_column_name] = ""
        for i in range(len(articles_df)):
            keywords_list = []
            for column in key_words_column_names:
                if type(articles_df[column].values[i]) == str:
                    articles_df[column].values[i] = articles_df[column].values[i].replace(",", ";")
                    articles_df[column].values[i] = ";".join([w for w in articles_df[column].values[i].split(";") if "full text" not in w.lower() and "article" not in w.lower() and w.strip() != ""])
                    keywords_list.extend(articles_df[column].values[i].split(";"))
                else:
                    keywords_list.extend(articles_df[column].values[i])
            keywords_list = [text_normalizer.normalize_sentence(key_word).lower() for key_word in keywords_list]
            articles_df[new_column_name].values[i] = [key_word for key_word in keywords_list if key_word != ""]
        return articles_df