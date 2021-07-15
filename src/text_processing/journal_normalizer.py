from text_processing import text_normalizer

class JournalNormalizer:

    def __init__(self):
        pass

    def normalize_journal_name(self, journal):
        journal = text_normalizer.remove_html_tags(journal)
        journal = text_normalizer.replace_brackets_with_signs(journal)
        if ";" in journal:
            separator_semicolon = False
            for val in journal.split(";"):
                if val.strip().lower().startswith("vol."):
                    separator_semicolon = True
            if separator_semicolon:
                journal = journal.split(";")[0]
            else:
                for val in journal.split(";"):
                    if "," in val:
                        journal = val.split(",")[0].strip()
                    else:
                        journal = val
        return journal.strip()

    def correct_journal_names(self, articles_df, journal_column = "journal"):
        for i in range(len(articles_df)):
            articles_df[journal_column].values[i] = self.normalize_journal_name(articles_df[journal_column].values[i])
        return articles_df