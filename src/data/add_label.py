import os
import pandas as pd
import numpy as np

# 1. تحديد المسارات بدقة بناءً على موقع السكربت
base_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_path = os.path.join(base_dir, '..', '..', 'data', 'processed', 'cleaned_lb.csv')
original_path = os.path.join(base_dir, '..', '..', 'data', 'raw', 'lb.csv')
output_path = os.path.join(base_dir, '..', '..', 'data', 'processed', 'labeled_lb.csv')

# تحميل البيانات الجديدة والقديمة (لجلب الـ Label)
df_cleaned = pd.read_csv(cleaned_path)
df_original = pd.read_csv(original_path) # نحتاجه فقط لعمود Priority

# 2. تجهيز المصفوفة y (الهدف) بناءً على حالة السيرفرات الفعلية
label_a = []
label_b = []

# سنعطي وزناً 70% للـ CPU و 30% للاتصالات لحساب الحمل الإجمالي
for index, row in df_cleaned.iterrows():
    score_a = (row['CPU Load A'] * 0.7) + (row['Conn A'] * 0.3)
    score_b = (row['CPU Load B'] * 0.7) + (row['Conn B'] * 0.3)
    
    # توجيه الطلب للسيرفر الأقل حملاً
    if score_a <= score_b:
        label_a.append(1)
        label_b.append(0)
    else:
        label_a.append(0)
        label_b.append(1)

# إضافة التسميات (Labels) كأعمدة جديدة
df_cleaned['Label_A'] = label_a
df_cleaned['Label_B'] = label_b

# 3. حفظ البيانات مع التسميات في ملف جديد
df_cleaned.to_csv(output_path, index=False)

print(f"X Shape: {df_cleaned.iloc[:, :8].shape}") # المتوقع (1254, 8)
print(f"y Shape: {df_cleaned[['Label_A', 'Label_B']].shape}")   # المتوقع (1254, 2)
print(f"\n✅ Labels added successfully! File saved at: {os.path.normpath(output_path)}")