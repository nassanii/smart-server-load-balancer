import os
import pandas as pd
import numpy as np

# 1. تحديد مسار الملف بدقة بناءً على موقع السكربت
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', '..', 'data', 'raw', 'lb.csv')

# تحميل الملف
df = pd.read_csv(file_path)

# 2. اختيار الأعمدة الضرورية فقط
# سنبقي على: Network Traffic, Request Size, Threshold, Response Time
required_columns = [
    'Network Traffic (MB/s)', 
    'Request Size (MB)', 
    'Threshold', 
    'Response Time (ms)'
]

df_cleaned = df[required_columns].copy()

# 3. توليد بيانات السيرفرات (CPU Load A, CPU Load B, Conn A, Conn B)
np.random.seed(42)  # لضمان بقاء البيانات ثابتة عند التجربة
server_health = np.random.rand(len(df_cleaned), 4)

# إضافة الأعمدة الجديدة للبيانات
df_cleaned.loc[:, 'CPU Load A'] = server_health[:, 0]
df_cleaned.loc[:, 'CPU Load B'] = server_health[:, 1]
df_cleaned.loc[:, 'Conn A'] = server_health[:, 2]
df_cleaned.loc[:, 'Conn B'] = server_health[:, 3]

# 4. تحويل البيانات إلى NumPy Array للبدء في العمل الرياضي
X_final = df_cleaned.values

print(f"Original Shape: {df.shape}")
print(f"Final Input Matrix Shape: {X_final.shape}")
print("\nFirst row (8 Features):")
print(X_final[0])

# 5. حفظ البيانات المنظفة في مجلد processed
processed_path = os.path.join(base_dir, '..', '..', 'data', 'processed', 'cleaned_lb.csv')
df_cleaned.to_csv(processed_path, index=False)
print(f"\nCleaned dataset with server health successfully saved to: {os.path.normpath(processed_path)}")