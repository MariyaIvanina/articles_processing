import pandas as pd
from interventions_labeling_lib.intervention_labels import InterventionLabels
import math
import os
from io import BytesIO
import base64

class ExcelWriter:

    def __init__(self):
        pass

    def save_data_in_excel(self, data, field_names, filename, column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30, column_width=[], column_coloured_headers=[],\
        dict_with_colors = InterventionLabels.INTERVENTION_LABELS_COLOR):
        df = pd.DataFrame(data, columns=field_names)
        self.save_df_in_excel(df, filename, column_interventions=column_interventions, column_probabilities=column_probabilities, column_outliers=column_outliers, width_column=width_column, \
            column_width=column_width, column_coloured_headers = column_coloured_headers, dict_with_colors = dict_with_colors)

    def get_header_format(self, writer, color = "#E26B0A"):
        format_header = writer.book.add_format()
        format_header.set_align('center')
        format_header.set_bold()
        format_header.set_bg_color(color)
        format_header.set_text_wrap()
        return format_header

    def save_df_in_excel(self, df, filename, sheet_name="Sheet 1", column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30,column_width=[], column_coloured_headers=[],\
        dict_with_colors = InterventionLabels.INTERVENTION_LABELS_COLOR):
        print('Saving...')
        writer = pd.ExcelWriter(filename, engine='xlsxwriter', 
            options={'strings_to_urls': False, 'strings_to_formulas': False})

        self.save_df(writer, df, filename, sheet_name=sheet_name, column_interventions=column_interventions, column_probabilities=column_probabilities, column_outliers=column_outliers, width_column=width_column,\
         column_width=column_width, column_coloured_headers = column_coloured_headers, dict_with_colors = dict_with_colors)

        writer.save()
        print('Saved to %s' % filename)

    def save_df(self, writer, df, filename, sheet_name, column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30, column_width=[], column_coloured_headers=[],\
     dict_with_colors = InterventionLabels.INTERVENTION_LABELS_COLOR):
        df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 0), header=True, encoding='utf8')

        for i in range(len(df.columns)):
            writer.sheets[sheet_name].set_column(i,i,width_column)

        for column in column_width:
            if column[0] in df.columns:
                col_index = list(df.columns).index(column[0])
                writer.sheets[sheet_name].set_column(col_index,col_index,column[1])

        format_header = self.get_header_format(writer)
        
        for col_num, value in enumerate(df.columns.values):
            writer.sheets[sheet_name].write(0, col_num, value, format_header)

        for columns, color in column_coloured_headers:
            for column in columns:
                col_index = list(df.columns).index(column)
                writer.sheets[sheet_name].write(0, col_index, column, self.get_header_format(writer, color = color))

        for column in column_interventions:
            if column in df.columns:
                ExcelFormatter.colorize_params(writer.book, writer.sheets[sheet_name], df, column, dict_with_colors = dict_with_colors)

        for column_info in column_outliers:
            if column_info[0] in df.columns:
                ExcelFormatter.highlight_value(writer.book, writer.sheets[sheet_name], df, column_info[0], column_info[1])
        
        for column in column_probabilities:
            if column in df.columns:
                ExcelFormatter.data_bar(writer.book, writer.sheets[sheet_name], df, column)

    def save_big_data_df_in_excel(self, df, folder, portion_number = 100000, sheet_name="Sheet 1", column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30, column_width=[], column_coloured_headers=[]):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for filename_id in range(math.ceil(len(df) /portion_number)):
            temp_df = pd.DataFrame(df.values[filename_id*portion_number:(filename_id+1)*portion_number], columns = df.columns)
            self.save_df_in_excel(temp_df, os.path.join(folder, "%d.xlsx"%filename_id), sheet_name = sheet_name, column_interventions=column_interventions, column_probabilities=column_probabilities, column_outliers=column_outliers, width_column=width_column, \
            column_width=column_width, column_coloured_headers = column_coloured_headers )

    def save_df_in_excel_in_separate_tabs(self, dfs, filename, sheet_names, column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30, column_width=[], column_coloured_headers=[]):
        print('Saving...')
        writer = pd.ExcelWriter(filename, engine='xlsxwriter', 
            options={'strings_to_urls': False, 'strings_to_formulas': False})

        for idx, df in enumerate(dfs):
            sheet_name = sheet_names[idx]
            self.save_df(writer, df, filename, sheet_name=sheet_name, column_interventions=column_interventions, column_probabilities=column_probabilities, column_outliers=column_outliers, width_column=width_column)

        writer.save()
        print('Saved to %s' % filename)

    def encode_excel_base64(self, dfs, sheet_names, column_interventions=[], column_probabilities=[], column_outliers=[], width_column=30, column_width=[], column_coloured_headers=[]):
        print('Saving...')
        bio = BytesIO()
        writer = pd.ExcelWriter(bio, engine='xlsxwriter', 
            options={'strings_to_urls': False, 'strings_to_formulas': False})

        for idx, df in enumerate(dfs):
            sheet_name = sheet_names[idx]
            self.save_df(writer, df, "", sheet_name=sheet_name, column_interventions=column_interventions, column_probabilities=column_probabilities, column_outliers=column_outliers, width_column=width_column,\
                column_width=column_width, column_coloured_headers=column_coloured_headers)

        writer.save()
        bio.seek(0)
        workbook = bio.read()
        return base64.b64encode(workbook)


class ExcelFormatter:
    COLORS = [
        ('#b8860b', 'white'),  # DarkGoldenRod
        ('#4682b4', 'white'),  # SteelBlue
        ('#e9967a', 'white'),  # DarkSalmon
        ('#87cefa', 'black'),  # LightSkyBlue
        ('#663399', 'white'),  # RebeccaPurple
        ('#008b8b', 'white'),  # DarkCyan
        ('#cd5c5c', 'white'),  # IndianRed
        ('#c71585', 'white'),  # MediumVioletRed
        ('#696969', 'white'),  # DimGrey
        ('#20b2aa', 'white'),  # LightSeaGreen
        ('#f4a460', 'black'),  # SandyBrown
        ('#6495ed', 'white'),  # CornflowerBlue
        ('#ff6347', 'white'),  # Tomato
        ('#add8e6', 'black'),  # LightBlue
        ('#8fbc8f', 'white'),  # DarkSeaGreen
        ('#00ced1', 'black'),  # DarkTurquoise
        ('#d2b48c', 'black'),  # Tan
        ('#006400', 'white'),  # DarkGreen
        ('#a0522d', 'white'),  # Sienna
        ('#ff7f50', 'black'),  # Coral
        ('#dc143c', 'white'),  # Crimson
        ('#ffd700', 'black'),  # Gold
        ('#db7093', 'white'),  # PaleVioletRed
    ]

    @staticmethod
    def data_bar(workbook, worksheet, df, column):
        col_index = list(df.columns).index(column)
        worksheet.conditional_format(1, col_index, len(df), col_index, {
            'type': 'data_bar',
            'min_value': df[column].min(),
            'max_value': df[column].max(),
            'min_type': 'num',
            'max_type': 'num',
        })

    @staticmethod
    def highlight_value(workbook, worksheet, df, column, value, bg_color='#ff7f50',font_color="black"):
        """
        Highlight certain value within a column
        """
        col_index = list(df.columns).index(column)
        worksheet.conditional_format(1, col_index, len(df), col_index, {
            'type': 'cell',
            'criteria': '==',
            'value': value if type(value) != str else '"%s"' % value,
            'format': workbook.add_format({
                'bg_color': bg_color, 'font_color': font_color
            })
        })

    @staticmethod
    def colorize_params(workbook, worksheet, df, column, dict_with_colors = InterventionLabels.INTERVENTION_LABELS_COLOR):
        """
        Colorize Intervention labels names
        """
        col_index = list(df.columns).index(column)
        for intervention_label in dict_with_colors:
            worksheet.conditional_format(1, col_index, len(df), col_index, {
                'type': 'cell',
                'criteria': '==',
                'value': '"%s"' % intervention_label,
                'format': workbook.add_format({
                    'bg_color': dict_with_colors[intervention_label][0],
                    'font_color': dict_with_colors[intervention_label][1]
                })
            })

