def send_file_via_email(email_subject, email_body, file_path, from_email="ngocdungxyz@gmail.com", to_email="ngocdungxyz@gmail.com"):
# Python code to illustrate Sending mail with attachments 
    # from your Gmail account 

    # libraries to be imported 
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 


    # instance of MIMEMultipart 
    msg = MIMEMultipart() 

    # storing the senders email address 
    msg['From'] = from_email

    # storing the receivers email address 
    msg['To'] = to_email

    # storing the subject 
    msg['Subject'] = email_subject

    # string to store the body of the mail 
    body = email_body

    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 

    # open the file to be sent 
    attachment = open(file_path, "rb") 

    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 

    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 

    # encode into base64 
    encoders.encode_base64(p) 

    p.add_header('Content-Disposition', "attachment") 

    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(from_email, "10179224") 

    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(from_email, to_email, text) 

    # terminating the session 
    s.quit() 
    
    
def send_image_via_email(email_subject, email_body, image_file_path, from_email="ozawamariajp@gmail.com", to_email="ozawamariajp@gmail.com"):
    """
    Send image via email
    """

    # libraries to be imported
    import os
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email.mime.image import MIMEImage
    from email import encoders


    # instance of MIMEMultipart 
    msg = MIMEMultipart() 

    # storing the senders email address 
    msg['From'] = from_email

    # storing the receivers email address 
    msg['To'] = to_email

    # storing the subject 
    msg['Subject'] = email_subject

    # string to store the body of the mail 
    body = email_body

    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 

    # open the image file to be sent 
    image_data = open(image_file_path, "rb").read() 
    
    # instance of MIMEImage and named as image
    image = MIMEImage(image_data, name=os.path.basename(image_file_path))

    # attach the image instance 'image' to instance 'msg' 
    msg.attach(image) 

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(from_email, "10179224") 

    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(from_email, to_email, text) 

    # terminating the session 
    s.quit()     
    
def send_images_via_email(email_subject, email_body, image_file_paths, sender_email="ozawamariajp@gmail.com", recipient_emails=["ozawamariajp@gmail.com"]):
    """
    Send image via email
    """

    # libraries to be imported
    import os
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email.mime.image import MIMEImage
    from email import encoders


    ### Create instance of MIMEMultipart message
    msg = MIMEMultipart() 

    # storing the senders email address 
    msg['From'] = sender_email

    ### Recipient emails
    msg['To'] = ", ".join(recipient_emails)

    ### SUBJECT
    msg['Subject'] = email_subject

    ### BODY TEXT
    # string to store the body of the mail 
    body = email_body
    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 
    
    ### ATTACH IMAGES
    for i_image in range(len(image_file_paths)):
        # open the image file to be sent 
        image_data = open(image_file_paths[i_image], "rb").read() 
        # instance of MIMEImage and named as image
        image = MIMEImage(image_data, name=os.path.basename(image_file_paths[i_image]))
        # attach the image instance 'image' to instance 'msg' 
        msg.attach(image) 

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(sender_email, "10179224") 

    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(sender_email, recipient_emails, text) 

    # terminating the session 
    s.quit()         
