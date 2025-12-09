# نستخدم نسخة بايثون جاهزة
FROM python:3.9

# نجهز فولدر العمل
WORKDIR /code

# تسطيب مكتبات النظام الضرورية للـ OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# نسخ ملف المكتبات وتسطيبها
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# نسخ باقي ملفات المشروع
COPY . /code

# منح صلاحيات للملفات (عشان ميديناش Error 403)
RUN chmod -R 777 /code

# تشغيل التطبيق على بورت 7860 (ده البورت بتاع Hugging Face)
CMD ["python", "app.py"]