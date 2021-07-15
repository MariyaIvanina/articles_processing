class USAIDInterventionLabels:

    INTERVENTION_LABELS_COLOR = {
        "Water infrastructure": ('#87cefa', 'black'),
        "Sanitation/Hygiene": ('#c71585', 'white'),
        "Fecal sludge management": ('#c71585', 'white'),
        "Menstrual hygiene management": ('#c71585', 'white'),
        "Water quality": ('#20b2aa', 'white'),
        "Human health": ('#20b2aa', 'black'),
        "Sustainability/Environmental health": ('#663399', 'white'),
        "Community and behavior": ('#70AD47', 'black'),
        "Assessment tool or program": ('#db7093', 'white'),
        "Policy": ('#dc143c', 'white'),
        "Agriculture": ('#696969', 'white'),
        "Non-intervention": ('#ffd700', 'black')
    }

    INTERVENTION_LABEL_TO_NUMBER = {
        "Water infrastructure": 1,
        "Sanitation/Hygiene": 2,
        "Fecal sludge management": 3,
        "Menstrual hygiene management": 4,
        "Water quality": 5,
        "Human health": 6,
        "Sustainability/Environmental health":7,
        "Community and behavior":8,
        "Assessment tool or program":9,
        "Policy": 10,
        "Agriculture": 11,
        "Non-intervention": 12
    }

    INTERVENTION_NUMBER_TO_LABEL = {
        1: "Water infrastructure",
        2: "Sanitation/Hygiene",
        3: "Fecal sludge management",
        4: "Menstrual hygiene management",
        5: "Water quality",
        6: "Human health",
        7: "Sustainability/Environmental health",
        8: "Community and behavior",
        9: "Assessment tool or program",
        10: "Policy",
        11: "Agriculture",
        12: "Non-intervention"
    }
