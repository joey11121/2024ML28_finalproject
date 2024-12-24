import os
import shutil
from music21 import *
import xml.etree.ElementTree as ET
import subprocess

class MusicXMLProcessor:
    def __init__(self, musescore_path=r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"):
        self.musescore_path = musescore_path
        
    def ensure_directory(self, directory):
        """確保目錄存在"""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def extract_part(self, xml_file, part_element, output_dir):
        """提取單個聲部並創建新的XML檔案"""
        try:
            # 創建新的XML文檔
            new_root = ET.Element('score-partwise', {'version': '4.0'})
            
            # 複製必要的元素
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 複製part-list但只保留當前part的資訊
            part_list = ET.SubElement(new_root, 'part-list')
            original_part_list = root.find('part-list')
            if original_part_list is not None:
                part_id = part_element.get('id')
                for score_part in original_part_list.findall('score-part'):
                    if score_part.get('id') == part_id:
                        part_list.append(ET.fromstring(ET.tostring(score_part)))
                        break
            
            # 添加part元素
            new_root.append(ET.fromstring(ET.tostring(part_element)))
            
            # 創建新的XML檔案
            part_id = part_element.get('id')
            output_file = os.path.join(output_dir, f"part_{part_id}.xml")
            
            # 寫入檔案
            tree = ET.ElementTree(new_root)
            tree.write(output_file, encoding='UTF-8', xml_declaration=True)
            
            return output_file
            
        except Exception as e:
            print(f"Error extracting part: {str(e)}")
            return None

    def convert_to_pdf(self, xml_file, output_file):
        """將XML檔案轉換為PDF"""
        try:
            command = [
                self.musescore_path,
                xml_file,
                '-o',
                output_file
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"MuseScore error: {result.stderr}")
                
            return output_file
            
        except Exception as e:
            print(f"Error converting to PDF: {str(e)}")
            return None

    def process_file(self, input_file, output_xml_dir, output_pdf_dir):
        """處理單個XML檔案，分割聲部並轉換為PDF"""
        try:
            # 確保輸出目錄存在
            self.ensure_directory(output_xml_dir)
            self.ensure_directory(output_pdf_dir)
            
            # 解析XML檔案
            tree = ET.parse(input_file)
            root = tree.getroot()
            
            # 處理每個part
            for part in root.findall('.//part'):
                part_id = part.get('id')
                print(f"Processing part {part_id}...")

                # 提取聲部，存放在單一目錄下
                xml_output = self.extract_part(input_file, part, output_xml_dir)
                if xml_output:
                    print(f"Created XML file: {xml_output}")
                    
                    # 轉換為PDF
                    pdf_output = os.path.join(output_pdf_dir, f"part_{part_id}.pdf")
                    if self.convert_to_pdf(xml_output, pdf_output):
                        print(f"Created PDF file: {pdf_output}")
                    else:
                        print(f"Failed to create PDF for part {part_id}")
                else:
                    print(f"Failed to extract part {part_id}")
                    
        except Exception as e:
            print(f"Error processing file: {str(e)}")

def main():
    # 設定路徑
    input_folder = "./split_output"
    output_xml_folder = "./output_xml"  # Unified directory for all XML parts
    output_pdf_folder = "./output_pdf"
    musescore_path = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
    
    # 創建處理器實例
    processor = MusicXMLProcessor(musescore_path)
    
    # 確保輸出目錄存在
    processor.ensure_directory(output_xml_folder)
    processor.ensure_directory(output_pdf_folder)
    
    # 處理輸入資料夾中的所有XML檔案
    xml_files = [f for f in os.listdir(input_folder) if f.endswith(('.xml', '.musicxml'))]
    
    for xml_file in xml_files:
        print(f"\nProcessing {xml_file}...")
        input_path = os.path.join(input_folder, xml_file)
        processor.process_file(input_path, output_xml_folder, output_pdf_folder)

if __name__ == "__main__":
    main()
