import smtplib
from email.message import EmailMessage

SENDER_EMAIL = "dymmy27@gmail.com"
APP_PASSWORD = "wnkncgxdhyfparuh"


print("Mail Start")
msg = EmailMessage()
msg['Subject'] = "Accident Event Detection"
msg['From'] = SENDER_EMAIL
msg['To'] = "narmadaugale27@gmail.com"
msg.set_content('Accident Event Detected')


with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)
        print("Mail send successfully")
