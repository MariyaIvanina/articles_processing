from sklearn.metrics import f1_score
from text_processing import text_normalizer
from interventions_labeling_lib import intervention_labeling
from interventions_labeling_lib import usaid_intervention_labels
import editdistance

def get_f1_score_test_data(test_data, intervention_labeler):
    res_pred, res_prob = intervention_labeler.predict_class(test_data.values, return_probs=True)
    res_true = test_data["Label"].values
    return f1_score(res_true, res_pred, average="macro")

def get_f1_multi_label(test_labels, y_true, max_label=12):
    final_res = []
    for i in range(len(test_labels)):
        identified_labels = set()
        for j in range(len(test_labels[0])):
            if test_labels[i][j] >= 0.5:
                identified_labels.add(j+1)
        if not identified_labels:
            identified_labels.add(max_label)
        final_res.append(identified_labels)
    print(len(final_res), len(y_true))
    cnt = 0
    cnt_precision = 0
    cnt_recall = 0
    cnt_f1 = 0
    cnt_all_found = 0
    for idx, res in enumerate(y_true):
        cnt += 1
        cnt_intersect = len(y_true[idx].intersection(final_res[idx]))
        cnt_correct = len(y_true[idx])
        cnt_found = len(final_res[idx])
        precision = 0 if cnt_found == 0 else (cnt_intersect/cnt_found)
        recall = 0 if cnt_correct == 0 else (cnt_intersect/cnt_correct)
        f1 = 0
        if (precision + recall) > 0:
            f1 = 2*precision*recall/(precision + recall)
        cnt_precision += precision
        cnt_recall += recall
        if recall > 0.99:
            cnt_all_found += 1
        cnt_f1 += f1
    return cnt_f1/cnt

def get_accuracy_multi_label(test_labels, y_true, max_label=12):
    final_res = []
    for i in range(len(test_labels)):
        identified_labels = set()
        for j in range(len(test_labels[0])):
            if test_labels[i][j] >= 0.5:
                identified_labels.add(j+1)
        if not identified_labels:
            identified_labels.add(max_label)
        final_res.append(identified_labels)
    cnt = 0
    cnt_precision = 0
    cnt_recall = 0
    cnt_f1 = 0
    cnt_all_found = 0
    for idx, res in enumerate(y_true):
        cnt += 1
        cnt_intersect = len(y_true[idx].intersection(final_res[idx]))
        cnt_correct = len(y_true[idx])
        cnt_found = len(final_res[idx])
        precision = 0 if cnt_found == 0 else (1 if cnt_intersect > 0 else 0)
        recall = 0 if cnt_correct == 0 else (1 if cnt_intersect > 0 else 0)
        f1 = 0
        if (precision + recall) > 0:
            f1 = 2*precision*recall/(precision + recall)
        cnt_precision += precision
        cnt_recall += recall
        if recall > 0.99:
            cnt_all_found += 1
        cnt_f1 += f1
    return cnt_f1/cnt

def normalize_full(text):
    return " ".join(sorted(text_normalizer.normalize_sentence(text).split(" ")))

def deduplicate(dfs):
    dicts = {}
    deduplicated_df_data = []
    for df in dfs:
        for i in range(len(df)):
            processed = normalize_full(df["Narrow concept"].values[i])
            if processed not in dicts:
                dicts[processed] = set()
                deduplicated_df_data.append(
                    (df["Narrow concept"].values[i], df["Broad concepts"].values[i], df["Label"].values[i]))
            dicts[processed].add(df["Label"].values[i])
    return dicts, pd.DataFrame(deduplicated_df_data, columns=["Narrow concept", "Broad concepts", "Label"])

def print_per_label_stats(test_res, res_label, _outcomes_sentence_labeler):
    from sklearn.metrics import f1_score
    f1_score_ = 0.0
    total_cnt = 0
    for idx, label in enumerate(labels):
        print(label)
        test_y = [1 if (idx+1) in test_res[i] else 0 for i in range(len(test_res))]
        res_y = [res_label[i][idx] for i in range(len(test_res))]
        _outcomes_sentence_labeler.print_summary(test_y, res_y)
        f1_score_ += f1_score(test_y, res_y, average="macro")
        total_cnt += 1
    print("Average ", f1_score_/total_cnt if total_cnt > 0 else 0)

def normalized_levenshtein_score(a, b):
    return 1 - editdistance.eval(a, b)/max(len(a), len(b))