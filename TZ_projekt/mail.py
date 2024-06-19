import smtplib
import os
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Nastavitve za prijavo
your_email = "jasa.rakun@gmail.com"
your_password = "clsq rtxi abts mmlf"

# Nastavitve prejemnika
recipient_email = "myfxpip@gmail.com"

# Nastavitev poti do mape z PDF-i
pdf_folder = r"C:\Users\jasar\Downloads"

# Ustvarjanje e-pošte
msg = MIMEMultipart()
msg['From'] = your_email
msg['To'] = recipient_email
msg['Subject'] = "Dnevni PDF-i"

# Poišči vse PDF-e, ustvarjene danes
today = datetime.date.today()
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file)
        file_time = datetime.date.fromtimestamp(os.path.getmtime(file_path))

        if file_time == today:
            part = MIMEBase('application', "octet-stream")
            part.set_payload(open(file_path, "rb").read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file_path))
            msg.attach(part)

# Pošiljanje e-pošte
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(your_email, your_password)
server.sendmail(your_email, recipient_email, msg.as_string())
server.quit()
