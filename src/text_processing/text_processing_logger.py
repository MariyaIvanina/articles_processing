import uuid
import os
import pickle

class TextProcessingLogger:

    def __init__(self, log_status_filename):
        self.log_status_filename = log_status_filename
        self.initialize_from_file()

    def get_initial_step(self, name):
        return {"step_name": name, "unique_id": uuid.uuid4().hex,"total_articles_count":0, "processed_articles_count":0, "percent_done":0, "status": "Pending", "errors":[]}

    def initialize_log_info(self, steps):
        steps_order = ["Text normalization", "Creating an inverted index search engine"]
        steps_info = {"Text normalization":self.get_initial_step("Text normalization"),\
        "Creating inverted index search engine":self.get_initial_step("Creating inverted index search engine")}
        for idx, step in enumerate(steps):
            new_name = "Keywords search for the dictionary from the file " + step["original_filename"]
            if new_name not in steps_info:
                steps_info[new_name] = self.get_initial_step(new_name)
                steps_order.append(new_name)
        self.info = {"steps_info": steps_info, "steps": steps_order,"status":"In Progress", "is_ready_to_download":False}
        self.save_info()

    def save_info(self):
        if self.log_status_filename.strip() != "":
            with open(self.log_status_filename, "wb") as f:
                pickle.dump(self.info, f)

    def initialize_from_file(self):
        if os.path.exists(self.log_status_filename):
            with open(self.log_status_filename, "rb") as f:
                self.info = pickle.load(f)
        else:
            self.info = {"steps_info": {}, "steps": [], "status":"In Progress", "is_ready_to_download":False}

    def update_status(self, status):
        self.initialize_from_file()
        self.info["status"] = status
        self.save_info()

    def update_status_for_step(self, step, status, errors = ""):
        self.initialize_from_file()
        if step in self.info["steps_info"]:
            errors = errors.strip()
            if errors != "" and errors not in self.info["steps_info"][step]["errors"]:
                self.info["steps_info"][step]["errors"].append(errors)
            if self.info["steps_info"][step]["status"]  != "Finished with errors":
                self.info["steps_info"][step]["status"] = status
        self.save_info()

    def update_step_results(self, step, processed_links, errors = ""):
        self.initialize_from_file()
        if step in self.info["steps_info"]:
            errors = errors.strip()
            self.info["steps_info"][step]["processed_articles_count"] += len(processed_links)
            if errors != "" and errors not in self.info["steps_info"][step]["errors"]:
                self.info["steps_info"][step]["errors"].append(errors)
        self.save_info()

    def perform_final_checks(self, is_ready_to_download = False):
        self.initialize_from_file()
        all_finished = True
        without_errors = True
        for query in self.info["steps_info"]:
            if not self.info["steps_info"][query]["status"] == "Finished" and not self.info["steps_info"][query]["status"] == "Finished with errors":
                all_finished = False
                break
            if self.info["steps_info"][query]["status"] == "Finished with errors":
                without_errors = False
        self.info["status"] = ("Finished" if without_errors else "Finished with errors") if all_finished else "In Progress"
        self.info["is_ready_to_download"] = is_ready_to_download
        self.save_info()

    def update_status_for_cancelling(self):
        self.initialize_from_file()
        self.info["status"] = "Cancelled" if self.info["status"] == "In Progress" else self.info["status"]
        for query in self.info["steps_info"]:
            self.info["steps_info"][query]["status"] = "Pending" if self.info["steps_info"][query]["status"] == "In Progress" else self.info["steps_info"][query]["status"]
        self.save_info()






