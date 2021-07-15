from text_processing import author_normalizer
from text_processing import author_affiliation_extractor
from text_processing import text_normalizer

class AuthorAndAffiliationsProcessing:

    def __init__(self, author_field = "author", affiliations_field = "affiliation"):
        self.author_normalizer = author_normalizer.AuthorNormalizer()
        self.author_affiliation_extractor = author_affiliation_extractor.AuthorAffiliationExtractor()
        self.author_field = author_field
        self.affiliations_field = affiliations_field

    def process_authors_and_affiliations(self, df, author_mapping = {}, affiliations_mapping_info = {}):
        authors, affiliations = [[] for i in range(len(df))], [[] for i in range(len(df))]
        if len(author_mapping) != 0:
            author_column = list(author_mapping.keys())[0]
            authors, affiliations = self.author_normalizer.normalize_authors(df[author_column].values)
            df[author_mapping[author_column]] = authors
        else:
            df[self.author_field] = authors
        raw_affiliations = ["" for i in range(len(df))]
        affiliation_column_name = self.affiliations_field
        if len(affiliations_mapping_info) != 0:
            affiliation_column = list(affiliations_mapping_info.keys())[0]
            raw_affiliations = df[affiliation_column].values
            affiliation_column_name = affiliations_mapping_info[affiliation_column]
        for i in range(len(df)):
            temp = raw_affiliations[i]
            for auth_affil in affiliations[i]:
                if auth_affil not in temp:
                    temp = temp + " ; " + auth_affil
            raw_affiliations[i] = text_normalizer.remove_html_tags(temp)
        self.author_affiliation_extractor.start_affiliation_remapping(raw_affiliations)
        df[affiliation_column_name] = ""
        for i in range(len(df)):
            if i % 3000 == 0 or i == len(df) -1:
                print("Processed %d articles"%i)
            df[affiliation_column_name].values[i] = self.author_affiliation_extractor.find_university_affiliations_from_string(raw_affiliations[i])
        return df           


