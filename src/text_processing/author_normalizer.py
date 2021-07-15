from text_processing import text_normalizer
import re

class AuthorNormalizer:

    def __init__(self):
        self.affiliation_markers =["universit", "institu", "college", "academy", "r&d", "foundation", "research center", "hochschul", \
                      "centre", "center", "organisat", "dept", "department", "world", "cooperation", "agriculture", "program", "office","committee",\
                      "project", "council", "international", "development", "training", "executive", "research", "division", "meeting",  "group", "national", "school"]
        self.abbreviations = ["CGIAR", "ICRISAT","UNDP","IIED","CIRAD","RARS","UNICEF","OECD","USDA","CIAT","IITA","CCAFS","ACIAR","EMBRAPA","IBPGR","CSIRO","UNEP","INRA",\
        "NWFP","ILRI", "AIDS", "IFAD", "ILCA", "CIFOR", "IUCN", 'UNESCO', "SADC", "USAID"]

    def normalize_raw_authors(self, authors_string):
        authors_string = self.replace_separators_for_authors(authors_string)
        new_authors = []
        author_affiliations = []
        for author in authors_string.split(";"):
            break_words = author.replace("(",",").replace("[",",").split(",")
            for idx, word_part in enumerate(break_words):
                if text_normalizer.contain_name(word_part.lower(), self.affiliation_markers) or text_normalizer.contain_name(word_part, self.abbreviations):
                    author_affiliations.append(",".join(break_words[idx:]).replace(")"," ").replace("]", " ").strip())
                    author = ",".join(break_words[:idx]).strip()
                    break
            author = re.sub("\(.*?\)|\[.*?\]", " ", author).replace(",", " ").strip()
            if len([w for w in author.split() if len(w.replace('.',"")) > 3]) > 4:
                author_affiliations.append(author)
                continue
            author_parts = list(filter(None, author.split()))
            if len(author_parts) > 0 and (len(author_parts[0]) < 3 or "." in author_parts[0]):
                author = (" ".join(author_parts[1:]) + " " + author_parts[0]).strip()
            author = text_normalizer.remove_copyright(author.replace('.', '').replace(',', '').replace(" ","").replace("-","").replace("_","").strip())
            if len(author) > 3 and "?" not in author:
                new_authors.append(author)
        return new_authors, author_affiliations

    def normalize_authors(self, authors):
        authors_all = []
        affiliations_all = []
        for i in range(len(authors)):
            new_authors, author_affiliations = self.normalize_raw_authors(authors[i])
            authors_all.append(new_authors)
            affiliations_all.append(author_affiliations)
        return authors_all, affiliations_all

    def find_full_author_names(self, word_string, separator = ";", reverse = False):
        filter_words = ["http", "anonym", "available", "session","various"]
        ind = -1 if reverse else 1
        pairs = []
        full_name = ""
        for word in [w.strip() for w in word_string.split(separator) if w.strip() != "" and not text_normalizer.contain_name(w.lower(), filter_words)][::ind]:
            if word.isupper() and (len(word) < 3 or ("." in word and " " not in word)):
                full_name = (full_name + " " + word) if not reverse else (word + " " + full_name)
            elif full_name != "":
                full_name = (full_name + " " + word) if not reverse else (word + " " + full_name)
                pairs.append(full_name.strip())
                full_name = ""
            else:
                pairs.append(word)
        if full_name != "":
            pairs.append(full_name.strip())
        return pairs[::ind]

    def replace_separators_for_authors(self, word_string):
        word_string = text_normalizer.remove_html_tags(word_string)
        word_string = re.sub(r"\b\d+(-?\d+)*\b", " ", word_string)
        separator = ";" if ";" in word_string else ","
        word_string = re.sub(r"\band\b", separator, word_string)
        word_string = re.sub(r"\b(authored|edited) (by)?\b", " ", word_string)
        word_string = re.sub(r"\b.*? named after\b", " ", word_string)
        word_string = re.sub(r"\b\\u.*?\b", " ", word_string)
        word_string = re.sub(r"\bet al\.?", " ", word_string).replace("=","").replace("\n"," ").strip()
        error = False
        res = self.find_full_author_names(word_string, separator = separator)
        for word in res:
            if len(word.replace(".","").replace(",","").strip()) < 3 or " " not in word:
                error = True
                break
        if error:
            error = False
            res = self.find_full_author_names(word_string, separator = separator, reverse = True)
            for word in res:
                if len(word.replace(".","").replace(",","").strip()) < 3 or " " not in word or len(word.split()) >3:
                    error = True
                    break
        if error:
            temp_word =  ""
            new_res = []
            for word in res:
                if " " not in word:
                    if word.isupper() and "." not in word:
                        if temp_word != "":
                            new_res.append(temp_word)
                        new_res.append(word)
                        temp_word = ""
                    else:
                        if temp_word == "":
                            temp_word = word
                        else:
                            new_res.append(temp_word + " " + word)
                            temp_word = ""
                else:
                    if temp_word != "":
                        new_res.append(temp_word + " " + word)
                    else:
                        new_res.append(word)
                    temp_word = ""
            if temp_word != "":
                new_res.append(temp_word)
            res = new_res
        return " ; ".join(res)