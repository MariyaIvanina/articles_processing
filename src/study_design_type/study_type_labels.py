class StudyTypeLabels:

    STUDY_TYPE_LABELS_COLOR = {
        "Observational study": ('#87cefa', 'black'),
        "Modeling study": ('#c71585', 'white'),
        "Laboratory study": ('#663399', 'white'),
        "Review paper": ('#006400', 'white'),
        "Field study": ('#ffd700', 'black'),
        "No category": ('#20b2aa', 'white')
    }

    STUDY_TYPE_LABEL_TO_NUMBER = {
        "Observational study":0,
        "Modeling study":1,
        "Laboratory study":2,
        "Review paper":3,
        "Field study":4,
        "No category":5
    }

    STUDY_TYPE_NUMBER_TO_LABEL = {
        0:"Observational study",
        1:"Modeling study",
        2:"Laboratory study",
        3:"Review paper",
        4:"Field study",
        5:"No category"
    }
