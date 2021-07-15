import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import math
import os
import pandas as pd
import re

def search_year(year, years):
    for idx, _year in enumerate(years):
        if idx == len(years) -1:
            continue
        if year >= _year and year < years[idx + 1]:
            return idx
    return -1

def save_plots_topics(folder, articles_df, column_name, topic_modeler, with_sorted = False,
        vmax = 800, relative_number = False, years = list(range(2008,2021)), cnt_per_plot = 25):
    if not os.path.exists(folder):
        os.makedirs(folder)
    topic_year = np.zeros((topic_modeler.n_components, len(years)-1), dtype = int if not relative_number else float)
    topic_map = {}
    topic_map_by_id = {}
    topic_id = 0
    all_articles_by_year = np.zeros(len(years)-1, dtype=int)
    for i in range(len(articles_df)):
        year_ind = search_year(articles_df["year"].values[i], years)
        if year_ind >= 0:
            for topic in articles_df[column_name].values[i]:
                if topic not in topic_map:
                    topic_map[topic] = topic_id
                    topic_map_by_id[topic_id] = topic
                    topic_id += 1
                topic_year[topic_map[topic]][year_ind] += 1
            all_articles_by_year[year_ind] += 1

    if with_sorted:
        result = sorted([(idx, topic_val) for idx,topic_val in enumerate(np.sum(topic_year, axis = 1))],key=lambda x: x[1], reverse = True)
    else:
        result = [(idx, topic_val) for idx,topic_val in enumerate(np.sum(topic_year, axis = 1))]
    if relative_number:
        topic_year /= all_articles_by_year
        topic_year *= 100
    
    for ind in range(math.ceil(topic_modeler.n_components/cnt_per_plot)):
        plt.figure(figsize=(15, 6), dpi=150)
        topic_year_df = pd.DataFrame(topic_year[[i for i, cnt in result[ind*cnt_per_plot:(ind+1)*cnt_per_plot]],:])
        topic_year_df.index = [ topic_map_by_id[i] for i, cnt in result[ind*cnt_per_plot:(ind+1)*cnt_per_plot]]
        topic_year_df.columns = [ "%d-%d"%(years[idx], years[idx+1]) for idx, year in enumerate(years) if idx != len(years) -1]
        if relative_number:
            ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax=vmax, annot=True, fmt=".1f")
        else:
            ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax=vmax, annot=True, fmt="d")
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dtopics.png'%(ind*cnt_per_plot+1, (ind+1)*cnt_per_plot)))

def save_plots_districts(folder, 
        big_dataset,countries = ["Nigeria/", "Malawi/", "Kenya/", "Tanzania/", "Mali/", "Zambia/", "Burkina Faso/", "Philippines/", "Bangladesh/"], with_sorted = False, image_format="eps"):
    for country in countries:
        country_folder = os.path.join(folder, country)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        districts_dict = {}
        districts_dict_interv = {}
        for i in range(len(big_dataset)):
            for district in big_dataset["districts"].values[i]:
                if country in district:
                    if district not in districts_dict:
                        districts_dict[district] = 0
                    districts_dict[district] += 1
                    if district not in districts_dict_interv:
                        districts_dict_interv[district] = {"technology intervention": 0, "socioeconomic intervention": 0, "ecosystem intervention": 0}
                    for column in ["technology intervention", "socioeconomic intervention", "ecosystem intervention"]:
                        if len(big_dataset[column].values[i]) > 0:
                            districts_dict_interv[district][column] += 1
        if with_sorted:
            result = sorted([(name, (interv_val["technology intervention"], interv_val["socioeconomic intervention"], interv_val["ecosystem intervention"]),\
                              sum(interv_val.values())) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
        else:
            result = sorted([(name, (districts_dict_interv[name]["technology intervention"], districts_dict_interv[name]["socioeconomic intervention"], districts_dict_interv[name]["ecosystem intervention"]),\
                              cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)

        for ind in range(math.ceil(len(districts_dict)/30)):
            plt.figure(figsize=(15, 6), dpi=150)
            topic_year_df = pd.DataFrame([val[1] for val in result[ind*30:(ind+1)*30]])
            topic_year_df.index = [val[0] for val in result[ind*30:(ind+1)*30]]
            topic_year_df.columns = ["Technology intervention", "Socioeconomic intervention", "Ecosystem intervention"]

            ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 50, annot=True, fmt = "d")
            plt.tight_layout()
            plt.savefig(os.path.join(country_folder,'%d-%dinterventions.%s'%(ind*30+1, (ind+1)*30, image_format)), format=image_format)


def save_plots_districts_unique(folder, big_dataset,countries = ["Nigeria/", "Malawi/", "Kenya/", "Tanzania/", "Mali/", "Zambia/", "Burkina Faso/", "Philippines/", "Bangladesh/"], with_sorted = False):
    for country in countries:
        country_folder = os.path.join(folder, country)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        districts_dict = {}
        districts_dict_interv = {}
        for i in range(len(big_dataset)):
            for district in big_dataset["districts"].values[i]:
                if country in district:
                    if district not in districts_dict:
                        districts_dict[district] = 0
                    districts_dict[district] += 1
                    if district not in districts_dict_interv:
                        districts_dict_interv[district] = {"technology intervention": set(), "socioeconomic intervention":set(), "ecosystem intervention": set()}
                    for column in ["technology intervention", "socioeconomic intervention", "ecosystem intervention"]:
                        for val in big_dataset[column].values[i]:
                            districts_dict_interv[district][column].add(val)
        if with_sorted:
            result = sorted([(name, (len(interv_val["technology intervention"]), len(interv_val["socioeconomic intervention"]), len(interv_val["ecosystem intervention"])),\
                              sum([len(interv_val[v]) for v in interv_val])) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
        else:
            result = sorted([(name, (len(districts_dict_interv[name]["technology intervention"]), len(districts_dict_interv[name]["socioeconomic intervention"]), len(districts_dict_interv[name]["ecosystem intervention"])),\
                              cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)

        for ind in range(math.ceil(len(districts_dict)/30)):
            plt.figure(figsize=(15, 6), dpi=150)
            topic_year_df = pd.DataFrame([val[1] for val in result[ind*30:(ind+1)*30]])
            topic_year_df.index = [val[0] for val in result[ind*30:(ind+1)*30]]
            topic_year_df.columns = ["Technology intervention", "Socioeconomic intervention", "Ecosystem intervention"]

            ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 50, annot=True, fmt = "d")
            plt.tight_layout()
            plt.savefig(os.path.join(country_folder,'%d-%dinterventions.png'%(ind*30+1, (ind+1)*30)))

def code_sequence(values):
    return 4*int("Technology intervention" in values) + 2*int("Socioeconomic intervention" in values) + int("Ecosystem intervention" in values)

def decode_sequence(num):
    values = []
    for idx, col in enumerate(["Eco", "Socio", "Tech"]):
        if num & 2**idx:
            values.append(col)
    return list(sorted(values))

def save_plots_districts_with_overlapping(folder, big_dataset,
        countries = ["Nigeria/", "Malawi/", "Kenya/", "Tanzania/", "Mali/", "Zambia/", "Burkina Faso/", "Philippines/", "Bangladesh/"], with_sorted = False, image_format="eps"):
    for country in countries:
        country_folder = os.path.join(folder, country)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)

        districts_dict = {}
        districts_dict_interv = {}
        for i in range(len(big_dataset)):
            for district in big_dataset["districts"].values[i]:
                if country in district:
                    if district not in districts_dict:
                        districts_dict[district] = 0
                    districts_dict[district] += 1
                    if district not in districts_dict_interv:
                        districts_dict_interv[district] = {}
                        for i in range(1,8):
                            districts_dict_interv[district][i] = 0
                    if code_sequence(big_dataset["intervention_labels"].values[i]) > 0:
                        districts_dict_interv[district][code_sequence(big_dataset["intervention_labels"].values[i])] += 1
        if with_sorted:
            result = sorted([(name, tuple([interv_val[w] for w in [4,2,1,6,3,5,7] ]),\
                              sum(interv_val.values())) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
        else:
            result = sorted([(name,  tuple([interv_val[w] for w in [4,2,1,6,3,5,7] ]),\
                              cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)

        for ind in range(math.ceil(len(districts_dict)/30)):
            plt.figure(figsize=(15, 6), dpi=150)
            topic_year_df = pd.DataFrame([val[1] for val in result[ind*30:(ind+1)*30]])
            topic_year_df.index = [val[0] for val in result[ind*30:(ind+1)*30]]
            topic_year_df.columns = ["; ".join(decode_sequence(w))for w in [4,2,1,6,3,5,7]]

            ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 50, annot=True, fmt = "d")
            plt.tight_layout()
            plt.savefig(os.path.join(country_folder,'%d-%dinterventions.%s'%(ind*30+1, (ind+1)*30, image_format)), format=image_format)

def save_plots_topics_interv(folder, articles_df, column_name, with_sorted = True, topic_numbers=125, image_format="eps"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    topic_year_names = {}
    topic_year = np.zeros(topic_numbers, dtype = int)
    topics_per_page = int(topic_numbers/5)
    for i in range(len(articles_df)):
        for topic in articles_df[column_name].values[i]:
            topic_num = int(re.search("#(\d+)", topic).group(1)) -1
            topic_year_names[topic_num] = topic
            topic_year[topic_num] += 1
    if with_sorted:
        result = sorted([(idx, topic_val) for idx,topic_val in enumerate(topic_year)],key=lambda x: x[1], reverse = True)
    else:
        result = [(idx, topic_val) for idx,topic_val in enumerate(topic_year)]
        
    for ind in range(5):
        plt.figure(figsize=(6, 6), dpi=150)
        topic_year_df = pd.DataFrame(topic_year[[i for i,cnt in result[ind*topics_per_page:(ind+1)*topics_per_page]]])
        topic_year_df.index = [ topic_year_names[i] for i,cnt in result[ind*topics_per_page:(ind+1)*topics_per_page]]
        topic_year_df.columns = ["All"]

        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", annot=True, fmt = "d", vmax = 50)
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dinterventions.%s'%(ind*topics_per_page+1, (ind+1)*topics_per_page, image_format)), format=image_format)

def save_plots_topics_cooccur(folder, articles_df, topic_num, column_name = "topics", with_sorted = True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    topic_year = np.zeros(150, dtype = int)
    for i in range(len(articles_df)):
        should_be_used = False
        
        for topic in articles_df[column_name].values[i]:
            _topic_num = int(re.search("#(\d+)", topic).group(1))
            if _topic_num == topic_num:
                should_be_used = True
        if should_be_used:
            for topic in articles_df[column_name].values[i]:
                _topic_num = int(re.search("#(\d+)", topic).group(1)) -1
                topic_year[_topic_num] += 1
    if with_sorted:
        result = sorted([(idx, topic_val) for idx,topic_val in enumerate(topic_year)],key=lambda x: x[1], reverse = True)
    else:
        result = [(idx, topic_val) for idx,topic_val in enumerate(topic_year)]
        
    for ind in range(5):
        plt.figure(figsize=(6, 6), dpi=150)
        topic_year_df = pd.DataFrame(topic_year[[i for i,cnt in result[ind*30:(ind+1)*30]]])
        topic_year_df.index = [ topic_year_names[i] for i,cnt in result[ind*30:(ind+1)*30]]
        topic_year_df.columns = ["All"]

        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", annot=True, fmt = "d", vmax = 50)
        plt.title("Coocurance of topics with the topic " + topic_year_names[topic_num-1])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dtopics.png'%(ind*30+1, (ind+1)*30)))

def save_plots_cooccur_interv(folder, big_dataset, topic_num, with_sorted = False, save_as_eps=False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    districts_dict = {}
    districts_dict_interv = {}
    for i in range(len(big_dataset)):
        should_be_used = False
        
        for topic in big_dataset["topics"].values[i]:
            _topic_num = int(re.search("#(\d+)", topic).group(1))
            if _topic_num == topic_num:
                should_be_used = True
        if should_be_used:
            for topic in big_dataset["topics"].values[i]:
                if topic not in districts_dict:
                    districts_dict[topic] = 0
                districts_dict[topic] += 1
                if topic not in districts_dict_interv:
                    districts_dict_interv[topic] = {"all":0, "technology intervention": 0, "socioeconomic intervention": 0, "ecosystem intervention": 0}
                for interv in big_dataset["Intervention labels"].values[i].split(";"):
                    districts_dict_interv[topic][interv] += 1
                districts_dict_interv[topic]["all"] += 1
    if with_sorted:
        result = sorted([(name, (interv_val["all"], interv_val["technology intervention"], interv_val["socioeconomic intervention"], interv_val["ecosystem intervention"]),\
                          interv_val["all"]) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
    else:
        result = sorted([(name, (districts_dict_interv[name]["all"],districts_dict_interv[name]["technology intervention"], districts_dict_interv[name]["socioeconomic intervention"], districts_dict_interv[name]["ecosystem intervention"]),\
                          cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)

    for ind in range(math.ceil(len(districts_dict)/30)):
        plt.figure(figsize=(15, 6), dpi=150)
        topic_year_df = pd.DataFrame([val[1] for val in result[ind*30:(ind+1)*30]])
        topic_year_df.index = [val[0] for val in result[ind*30:(ind+1)*30]]
        topic_year_df.columns = ["All","Technology interv.", "Socioeconomic interv.", "Ecosystem interv."]

        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 50, annot=True, fmt = "d")
        plt.title("Coocurance of topics with the topic " + topic_year_names[topic_num-1])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dinterventions.png'%(ind*30+1, (ind+1)*30)))

def save_plots_cooccur_interv_relative(folder, big_dataset, topic_num, with_sorted = False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    districts_dict = {}
    districts_dict_interv = {}
    for i in range(len(big_dataset)):
        should_be_used = False
        
        for topic in big_dataset["topics"].values[i]:
            _topic_num = int(re.search("#(\d+)", topic).group(1))
            if _topic_num == topic_num:
                should_be_used = True
        if should_be_used:
            for topic in big_dataset["topics"].values[i]:
                if topic not in districts_dict:
                    districts_dict[topic] = 0
                districts_dict[topic] += 1
                if topic not in districts_dict_interv:
                    districts_dict_interv[topic] = {"all":0, "technology intervention": 0, "socioeconomic intervention": 0, "ecosystem intervention": 0}
                for interv in big_dataset["Intervention labels"].values[i].split(";"):
                    districts_dict_interv[topic][interv] += 1
                districts_dict_interv[topic]["all"] += 1
    if with_sorted:
        result = sorted([(name, (interv_val["technology intervention"]*100/interv_val["all"], interv_val["socioeconomic intervention"]*100/interv_val["all"], interv_val["ecosystem intervention"]*100/interv_val["all"]),\
                          interv_val["all"]) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
    else:
        result = sorted([(name, (districts_dict_interv[name]["technology intervention"]*100/districts_dict_interv[name]["all"], districts_dict_interv[name]["socioeconomic intervention"]*100/districts_dict_interv[name]["all"], districts_dict_interv[name]["ecosystem intervention"]*100/districts_dict_interv[name]["all"]),\
                          cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)
    
    for ind in range(math.ceil(len(districts_dict)/30)):
        plt.figure(figsize=(15, 6), dpi=150)
        topic_year_df = pd.DataFrame([val[1] for val in result[ind*30:(ind+1)*30]])
        topic_year_df.index = [val[0] for val in result[ind*30:(ind+1)*30]]
        topic_year_df.columns = ["Technology interv.", "Socioeconomic interv.", "Ecosystem interv."]

        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 80, annot=True, fmt = "0.1f")
        plt.title("Coocurance of topics with the topic " + topic_year_names[topic_num-1])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dinterventions.png'%(ind*30+1, (ind+1)*30)))

def save_plots_interventions_districts(folder, big_dataset, column_name= "Intervention labels", with_sorted = False, image_format="eps"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    districts_dict = {}
    districts_dict_interv = {}
    for i in range(len(big_dataset)):
        for topic in big_dataset["topics"].values[i]:
            if topic not in districts_dict:
                districts_dict[topic] = 0
            districts_dict[topic] += 1
            if topic not in districts_dict_interv:
                districts_dict_interv[topic] = {"technology intervention": 0, "socioeconomic intervention": 0, "ecosystem intervention": 0}
            for interv in big_dataset[column_name].values[i]:
                interv = interv.lower()
                if interv in districts_dict_interv[topic]:
                    districts_dict_interv[topic][interv] += 1
    if with_sorted:
        result = sorted([(name, (interv_val["technology intervention"], interv_val["socioeconomic intervention"], interv_val["ecosystem intervention"]),\
                          sum(interv_val.values())) for name,interv_val in districts_dict_interv.items()],key=lambda x: x[2], reverse = True)
    else:
        result = sorted([(name, (districts_dict_interv[name]["technology intervention"], districts_dict_interv[name]["socioeconomic intervention"], districts_dict_interv[name]["ecosystem intervention"]),\
                          cnt) for name, cnt in districts_dict.items()], key = lambda x: x[2], reverse= True)

    for ind in range(math.ceil(len(districts_dict)/25)):
        plt.figure(figsize=(15, 6), dpi=150)
        topic_year_df = pd.DataFrame([val[1] for val in result[ind*25:(ind+1)*25]])
        topic_year_df.index = [val[0] for val in result[ind*25:(ind+1)*25]]
        topic_year_df.columns = ["Technology intervention", "Socioeconomic intervention", "Ecosystem intervention"]

        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax = 50000, annot=True, fmt = "d")
        plt.tight_layout()
        plt.savefig(os.path.join(folder,'%d-%dinterventions.%s'%(ind*25+1, (ind+1)*25, image_format)), format=image_format)

def save_population_vs_geo_regions(folder, articles_df, vmax = 800, relative_number = False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    geo_regions = list(set([geo_reg  for geo_region in articles_df["geo_regions"] for geo_reg in geo_region]))
    geo_regions_vs_population = np.zeros((len(geo_regions), 3), dtype = int if not relative_number else float)
    all_articles_by_type = np.zeros(len(geo_regions), dtype=int)
    for i in range(len(articles_df)):
        for geo_region in  articles_df["geo_regions"].values[i]:
            if geo_region in geo_regions:
                geo_ind = geo_regions.index(geo_region)
                if "Small scale farmers" in articles_df["population tags"].values[i]:
                    geo_regions_vs_population[geo_ind][0] += 1
                elif "Farmers" in articles_df["population tags"].values[i]:
                    geo_regions_vs_population[geo_ind][1] += 1
                else:
                    geo_regions_vs_population[geo_ind][2] += 1
                all_articles_by_type[geo_ind] += 1

    
    result = [(idx, cnt_val) for idx,cnt_val in enumerate(np.sum(geo_regions_vs_population, axis = 1))]
    if relative_number:
        geo_regions_vs_population = geo_regions_vs_population.T
        geo_regions_vs_population /= all_articles_by_type
        geo_regions_vs_population *= 100
        geo_regions_vs_population = geo_regions_vs_population.T

    plt.figure(figsize=(15, 6), dpi=150)
    topic_year_df = pd.DataFrame(geo_regions_vs_population[[i for i,cnt in result],:])
    topic_year_df.index = geo_regions
    topic_year_df.columns = ["Small scale farmers", "Farmers", "Undefined"]
    if relative_number:
        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax=vmax, annot=True, fmt=".1f")
    else:
        ax = sns.heatmap(topic_year_df, linewidth=0.5, cmap="YlGnBu", vmin = 0, vmax=vmax, annot=True, fmt="d")
    plt.tight_layout()
    plt.savefig(os.path.join(folder,'plot.png'))

def run_plots():
	save_plots_topics("topics_climate_relative_rearranged", subset_df, "topics", topic_modeler, with_sorted = True, vmax = 20, relative_number=True)
	save_plots_topics("topics_climate_relative", subset_df, "topics", topic_modeler, with_sorted = False, vmax = 20, relative_number=True)
	save_plots_topics("topics_up_to_date_125", big_dataset, "topics", topic_modeler, with_sorted = False, vmax = 3000)
	save_plots_topics("topics_up_to_date_rearranged_125", big_dataset, "topics", topic_modeler, with_sorted = True, vmax = 3000)
	save_plots_topics("topics_up_to_date_125_relative_number", big_dataset, "topics", topic_modeler, with_sorted = False, vmax = 15,relative_number=True)
	save_plots_topics("topics_up_to_date_rearranged_125_relative_number", big_dataset, "topics", topic_modeler, with_sorted = True, vmax = 15,relative_number=True)
	save_plots_topics("topics_climate_subset", subset_df, "topics_new", topic_modeler, with_sorted = False)
	save_plots_topics("topics_climate_relative_subset_rearranged", subset_df, "topics_new", topic_modeler, with_sorted = True,vmax = 20, relative_number = True)
	save_plots_topics("topics_climate_relative_subset", subset_df, "topics_new", topic_modeler, with_sorted = False,vmax = 20, relative_number = True)
	save_plots_topics("topics_climate_rearranged_subset", subset_df, "topics_new", topic_modeler, with_sorted = True)
	save_plots_districts_with_overlapping("countries_plots_with_overlapping", big_dataset, with_sorted=True)
	save_plots_districts_unique("countries_plots_unique", big_dataset, with_sorted=True)
	save_plots_districts("countries_plots", big_dataset, with_sorted=True)
	save_plots_topics_interv("topic_interventions", temp_df, "topics", with_sorted = True)
	for topic in [30, 81, 121, 140, 10, 25, 91, 112, 124, 97]:
	    save_plots_cooccur_interv_relative("topic_coocur_interv_relative_topic_%d"%topic, all_df, topic, with_sorted=True)
	    save_plots_cooccur_interv("topic_coocur_interv_topic_%d"%topic, all_df, topic, with_sorted=True)
	    save_plots_topics_cooccur("topic_coocur_topic_%d"%topic, all_df, topic, with_sorted=True)
	save_plots_cooccur_interv("topic_coocur_weather_interv_topic_97", all_df, 97, with_sorted=True)
	save_plots_topics_cooccur("topic_coocur_ICT_topic_111", all_df, 111, with_sorted = True)
	save_plots_interventions_districts("intervention_labels_vs_topics", all_df, with_sorted = False)