from interventions_labeling_lib import hyponym_search
from interventions_labeling_lib import hearst_pattern_finder
from interventions_labeling_lib import hyponym_statistics
from text_processing import text_normalizer
from scipy.spatial import distance
import numpy as np
from interventions_labeling_lib import intervention_labeling
from time import time
from utilities import excel_writer
import pickle
import pandas as pd
import re
import os

class InterventionsPairsSearch:
    
    def __init__(self, key_word_mappings):
        self.key_word_mappings = key_word_mappings
    
    def create_pairs(self, sentences):
        res = []
        for expressions,sent in sentences:
            interventions = []
            narrow_concepts = []
            for word in expressions:
                is_intervention = False
                for key_word in self.key_word_mappings:
                    if key_word in word:
                        is_intervention = True
                if is_intervention:
                    interventions.append(word)
                narrow_concepts.append(word)
            if len(interventions) > 0:
                res.append([(narrow_concept, ";".join(interventions),sent,0) for narrow_concept in narrow_concepts])
        return res

class SentenceWithInterventionsLabeler:

	def __init__(self, filter_words):
		self.filter_words = filter_words
		self.load_labeled_data()
		self.id2title = {}

	def save_to_file(self, filename):
		pickle.dump(self.hyponyms_search_part, open(filename,"wb"))
		pickle.dump(self.id2title, open(filename.replace(".pickle","_2.pickle"),"wb"))

	def parse_sentences(self, articles_df, search_engine_inverted_index, key_word_mappings, parse=True, filename="../model/sentence_parsing_sm.pickle"):
		self.hs = hearst_pattern_finder.HearstPatterns()
		if parse:
			self.hyponyms_search_part = hyponym_search.HyponymsSearch()
			try:
			    hyponyms_search_part = pickle.load(open(filename,"rb"))
			    for i in hyponyms_search_part.global_hyponyms:
			    	self.hyponyms_search_part.add_hyponyms(hyponyms_search_part.global_hyponyms[i],i)
			    self.id2title = pickle.load(open(filename.replace(".pickle","_2.pickle"),"rb"))
			except:
			    self.id2title = {}
			interventions_pair_search = InterventionsPairsSearch(key_word_mappings)
			for i in range(len(articles_df)):
			    if i in self.id2title:
			        continue
			    self.id2title[i] = articles_df["title"].values[i]
			    if i % 5000 == 0 or i == len(articles_df) - 1:
			        self.save_to_file(filename)
			        print("Processed %d articles"%i)
			    for pairs in interventions_pair_search.create_pairs(self.hs.find_interventions_pairs(articles_df["title"].values[i] + (" " if len(articles_df["title"].values[i].strip()) == 0 or articles_df["title"].values[i].strip()[-1] in [".","?","!"] else ". ") + articles_df["abstract"].values[i])):
			        self.hyponyms_search_part.add_hyponyms(pairs,i)
			self.hyponyms_search_part.create_hyponym_dict(search_engine_inverted_index)
			print("Merged hyponyms")
			self.save_to_file(filename)
		else:
			self.hyponyms_search_part, self.id2title = pickle.load(open(filename,"rb"))
		self.hs_stat = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index, self.hyponyms_search_part.dict_hyponyms, self.hyponyms_search_part, filter_word_list = self.filter_words)

	def softmax(self, x):
	    e_x = np.exp(x - np.max(x))
	    e_x = e_x / e_x.sum(axis=0)
	    return e_x

	def get_normalized_distances(self, vec):
	    return (vec - np.mean(vec))/np.std(vec)

	def predict_class(self, rows, print_prob = False, folder = "..\\model\\sentence_intervention_labels_model"):
	    intervention_labeler = intervention_labeling.InterventionLabeling()
	    intervention_labeler.load_previous_models(folder)
	    predictions = intervention_labeler.predict_class(rows, give_probabilities= True)
	    probabilities = [self.softmax(self.get_normalized_distances(data)) for data in predictions]
	    if print_prob:
	        print(probabilities)
	    class_prob = np.argmax(probabilities,axis=1)
	    class_prob_sort = [np.argsort(data)[::-1] for data in probabilities]
	    class_thresholds = {0:0.4, 1:0.4, 2:0.5,3:0.4, 4:0.4}
	    return [class_prob[i] + 1  if probabilities[i][class_prob[i]] > class_thresholds[class_prob[i]] else 5 for i in range(len(class_prob))]

	def get_majority_class(self, classes):
	    classes_dict = {}
	    for cl in classes:
	        if cl not in classes_dict:
	            classes_dict[cl] = 0
	        classes_dict[cl] += 1
	    max_vote = 0
	    max_class = 5
	    for cl in [2,1,3,4]:
	        if cl in classes_dict and classes_dict[cl] >= max_vote:
	            max_vote = classes_dict[cl] 
	            max_class = cl
	    return max_class

	def get_all_classes(self, classes):
	    ans = set()
	    for cl in classes:
	        if cl != 5:
	            ans.add(cl)
	    return ans if len(ans) > 0 else set([5])

	def get_doc_sentences(self, interventions):
	    dict_with_sentences = {}
	    for pair in interventions:
	        if pair[2] not in dict_with_sentences:
	            dict_with_sentences[pair[2]] = []
	        if pair[0] in self.hyponyms_search_part.new_mapping and pair[1] in self.hyponyms_search_part.new_mapping:
	            dict_with_sentences[pair[2]].append({self.hyponyms_search_part.new_mapping[pair[0]]:[self.hyponyms_search_part.new_mapping[pair[1]]]})
	    return dict_with_sentences

	def debug_sentence_labeling(self, articles_number, folder = "..\\model\\sentence_intervention_labels_model"):
		doc_sentences = self.get_doc_sentences(self.hyponyms_search_part.global_hyponyms[articles_number])
		for sent in doc_sentences:
		    results = []
		    for pairs in doc_sentences[sent]:
		        processed_data = self.hs_stat.get_pruned_pairs_with_info(self.hs_stat.get_pruned_pairs(pairs), 0.3, True)
		        if len(processed_data) > 0:
		            processed_data = np.asarray(processed_data, dtype=object)
		            print(processed_data)
		            class_pred = self.predict_class(processed_data, print_prob = True, folder = folder)[0]
		            print(class_pred)
		            results.append(class_pred)
		    print(sent)
		    print(self.get_all_classes(results))

	def label_sentences(self,filename, folder = "..\\model\\sentence_intervention_labels_model"):
		res = []
		processed = []
		sent_dict = {}
		sent_num = 0
		start = time()
		for i,doc in enumerate(self.hyponyms_search_part.global_hyponyms):
		    if i % 50 == 0:
		        print("Processed %d documents" %(i) )
		    doc_sentences = self.get_doc_sentences(self.hyponyms_search_part.global_hyponyms[doc])
		    for sent in doc_sentences:
		        sent_dict[sent_num] = sent
		        for pairs in doc_sentences[sent]:
		            processed_data = self.hs_stat.get_pruned_pairs_with_info(self.hs_stat.get_pruned_pairs(pairs),0.3,True)
		            if len(processed_data) > 0:
		                processed_data = np.asarray(processed_data, dtype=object)
		                new_row = list(processed_data[0])
		                new_row.append(doc)
		                new_row.append(sent_num)
		                processed.append(tuple(new_row))
		        sent_num += 1
		print("Start processing probabilities %0.3f"%(time() - start))
		start = time()
		pred_classes = self.predict_class(processed, folder = folder)
		print("Finished processing probabilities %0.3f"%(time() - start))
		start = time()
		doc_to_sent = {}
		for i in range(len(pred_classes)):
		    if processed[i][-2] not in doc_to_sent:
		        doc_to_sent[processed[i][-2]] = {}
		    if processed[i][-1] not in doc_to_sent[processed[i][-2]]:
		        doc_to_sent[processed[i][-2]][processed[i][-1]] = []
		    doc_to_sent[processed[i][-2]][processed[i][-1]].append(pred_classes[i])
		for doc in doc_to_sent:
		    for sent_id in doc_to_sent[doc]:
		        sentence_res = self.get_all_classes(doc_to_sent[doc][sent_id])
		        sent_doc = sent_dict[sent_id].replace("NP_", " ").replace("_", " ").replace("#",":").replace("@"," ")
		        sentence_normalized = text_normalizer.normalize_sentence(sent_doc)
		        res.append((self.id2title[doc] if doc in self.id2title else "", sent_doc, self.join_result_to_string(sentence_res), \
		        	(self.join_result_to_string(self.class_dict[sentence_normalized]) if sentence_normalized in self.class_dict else np.nan)))
		print("Finished class prediction %0.3f"%(time() - start))
		start = time()
		writer = excel_writer.ExcelWriter()
		writer.save_data_in_excel(res, ["Article Title","Sentence","Label", "Label J"], filename)
		print("Finished %0.3f"%(time() - start))
		self.calculate_acc(res)

	def join_result_to_string(self, res):
		return ";".join([str(s) for s in res])

	def load_labeled_data(self):
		try:
			temp_df = pd.read_excel("../notebooks/JP_labeled.xlsx")
			self.class_dict = {}
			for i in range(len(temp_df)):
			    if temp_df["Label J"].values[i] == temp_df["Label J"].values[i]:
			        sentence_normalized = text_normalizer.normalize_sentence(temp_df["Sentence"].values[i])
			        if type(temp_df["Label J"].values[i]) != str:
			            self.class_dict[sentence_normalized] = set([int(temp_df["Label J"].values[i])])
			        else:
			            self.class_dict[sentence_normalized] = set([int(label) for label in temp_df["Label J"].values[i].split(";")])
		except:
			self.class_dict = {}

	def calculate_acc(self, res):
	    count = 0
	    precision = 0
	    recall = 0
	    for r in res:
	        sentence_normalized = text_normalizer.normalize_sentence(r[1])
	        if sentence_normalized in self.class_dict:
	            all_res = [int(part) for part in r[2].split(";")]
	            precision += len(self.class_dict[sentence_normalized].intersection(all_res))/ len(all_res)
	            recall += len(self.class_dict[sentence_normalized].intersection(all_res))/ len(self.class_dict[sentence_normalized])
	            count += 1
	    precision = precision/count
	    recall = recall/count
	    if count > 0:
	        print("Results ",count, precision, recall, 2*precision*recall/(precision+recall))