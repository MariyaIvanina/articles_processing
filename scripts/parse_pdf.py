from PIL import Image 
import pytesseract 
import sys 
from pdf2image import pdfinfo_from_path, convert_from_path
import os
import argparse
import easyocr
import pandas as pd
from time import time
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--file_path', default="")
    parser.add_argument('--folder_path', default="")
    parser.add_argument('--method', default="tesseract")
    parser.add_argument('--save_method', default="excel")
    parser.add_argument('--folder_to_save', default="")
    args = parser.parse_args()
  

    print("File to pdf: ", args.file_path)
    print("Folder for pdf extracting: ", args.folder_path)
    print("Method(tesseract or easyocr): ", args.method)
    print("Save method(excel or txt): ", args.save_method)
    print("Folder to save: ", args.folder_to_save)

    if args.folder_to_save:
        os.makedirs(args.folder_to_save, exist_ok=True)
    
    filenames = []
    if args.file_path:
        filenames.append(args.file_path)
    if args.folder_path:
        filenames.extend([os.path.join(args.folder_path, file) for file in os.listdir(args.folder_path)])

    for PDF_file in filenames:
        try:
            print(PDF_file)
            
            start_time = time()
            tmp_folder = os.path.basename(PDF_file).split(".")[0]
            filename_to_save = tmp_folder + ".xlsx"
            if args.folder_to_save:
                filename_to_save = os.path.join(args.folder_to_save, filename_to_save)

            if os.path.exists(filename_to_save):
                print("Already processed ", filename_to_save)
                continue

            os.makedirs(tmp_folder, exist_ok=True)
            
            info = pdfinfo_from_path(PDF_file, userpw=None, poppler_path=None)

            maxPages = info["Pages"]
            image_counter = 1
            print("Num of pages: ", maxPages)
            for page in range(1, maxPages + 1, 5) : 
                pages = convert_from_path(PDF_file, dpi=200, first_page=page, last_page = min(page+5-1, maxPages))

                for page in pages:  
                    filename = "page_" + str(image_counter) + ".jpg"
                    page.save(os.path.join(tmp_folder, filename), 'JPEG')
                    image_counter = image_counter + 1

            pdf_df_data = []
              
            if args.save_method == "txt":
                outfile = "out_text_%s_%s.txt"%(tmp_folder, args.method)
                f = open(outfile, "a")

            easyocr_reader = None
            if args.method == "easyocr":
                easyocr_reader = easyocr.Reader(['en']) # need to run only once to load model into memory

            for i in range(1, image_counter): 
                print(i)
                filename = "page_"+str(i)+".jpg"
                filename = os.path.join(tmp_folder, filename)
                      

                if args.method == "tesseract":
                    text = str(((pytesseract.image_to_string(Image.open(filename), config=" --psm 1"))))
                    text = text.replace('-\n', '')
                elif args.method == "easyocr":
                    text = "\n".join(easyocr_reader.readtext(filename, detail = 0))

                if args.save_method == "txt":
                    f.write(text)
                    f.write("### Page %d ###\n"%i)

                if args.save_method == "excel":
                    pdf_df_data.append((PDF_file, i, text))
                """
                if args.save_method == "excel":
                    pdf_df = pd.DataFrame(pdf_df_data, columns=["Filename", "Page", "Text"])
                    writer = pd.ExcelWriter(filename_to_save.replace(".xlsx", "_%d.xlsx"%i), engine='xlsxwriter', 
                        options={'strings_to_urls': False, 'strings_to_formulas': False})
                    pdf_df.to_excel(writer, index=False, freeze_panes=(1, 0), header=True, encoding = "utf-8")
                    writer.save()
                """

            if args.save_method == "txt":
                f.close()
            if args.save_method == "excel":
                pdf_df = pd.DataFrame(pdf_df_data, columns=["Filename", "Page", "Text"])
                writer = pd.ExcelWriter(filename_to_save, engine='xlsxwriter', 
                    options={'strings_to_urls': False, 'strings_to_formulas': False})
                pdf_df.to_excel(writer, index=False, freeze_panes=(1, 0), header=True, encoding = "utf-8")
                writer.save()
            try:
                shutil.rmtree(tmp_folder)
            except Exception as err:
                print("Didn't delete folder ", tmp_folder)
                print(err)
            print("Processed time: ", time() - start_time)
        except Exception as err:
            print("Didn't process ", PDF_file)
            print(err)
