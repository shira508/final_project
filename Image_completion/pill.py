import os
from PIL import Image

def convert_folder_to_grayscale(input_folder, output_folder):
    # יצירת תיקיית היעד אם היא לא קיימת
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"נוצרה תיקייה חדשה בשם: {output_folder}")

    # מעבר על כל הקבצים בתיקיית המקור
    for filename in os.listdir(input_folder):
        # בדיקה שהקובץ הוא אכן תמונה (לפי הסיומת)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # פתיחת התמונה
                with Image.open(input_path) as img:
                    # המרה לשחור-לבן (ערוץ L מייצג Luminance / גרייסקייל)
                    grayscale_img = img.convert('L')
                    
                    # שמירת התמונה החדשה בתיקיית היעד
                    grayscale_img.save(output_path)
                    print(f"הומר בהצלחה: {filename}")
            except Exception as e:
                print(f"שגיאה בעיבוד הקובץ {filename}: {e}")

# --- הגדרת הנתיבים שלך ---
# החליפי את הנתיבים האלו לנתיבים האמיתיים אצלך במחשב או בדרייב
input_directory = r"C:\Projects\FinalProject\NewProject\dataset\damaged"
output_directory = r"C:\Projects\FinalProject\NewProject\dataset\broke"

# הפעלת הפונקציה
convert_folder_to_grayscale(input_directory, output_directory)