class InterventionLabels:

    INTERVENTION_LABELS_COLOR = {
        "Technology intervention": ('#87cefa', 'black'),
        "Socioeconomic intervention": ('#c71585', 'white'),
        "Ecosystem intervention": ('#20b2aa', 'white'),
        "Storage intervention": ('#663399', 'white'),
        "Mechanisation intervention": ('#006400', 'white'),
        "Non-intervention": ('#ffd700', 'black')
    }

    INTERVENTION_LABEL_TO_NUMBER = {
        "Technology intervention": 1,
        "Socioeconomic intervention": 2,
        "Ecosystem intervention": 3,
        "Storage intervention":4,
        "Mechanisation intervention":5,
        "Non-intervention": 6
    }

    INTERVENTION_NUMBER_TO_LABEL = {
        1: "Technology intervention",
        2: "Socioeconomic intervention",
        3: "Ecosystem intervention",
        4: "Storage intervention",
        5: "Mechanisation intervention",
        6: "Non-intervention"
    }
