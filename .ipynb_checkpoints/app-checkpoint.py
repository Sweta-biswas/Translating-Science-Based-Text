from flask import Flask, render_template, request, jsonify, session, url_for, redirect,send_file,make_response
from pymongo import MongoClient
import certifi
import torch
import re
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import bcrypt
import pyautogui
import img2pdf
import os

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

app = Flask(__name__)


app.config["SECRET_KEY"] = "1241f366ecf2af7cbf180a0bab94fbdea617358a"
MONGO_URI = "mongodb+srv://admin:ScienceSubjectTranslation@cluster0.ydvb0ch.mongodb.net/ScienceSubjectTranslation?retryWrites=true&w=majority&appName=Cluster0"

tls_ca_file = certifi.where()

client = MongoClient(MONGO_URI, tlsCAFile=tls_ca_file)
db = client.ScienceSubjectTranslation

@app.route('/')
def index():
    if 'email' in session:
        email = session['email']
        return render_template('index2.html', email=email)
    else:
        return render_template('index.html')




@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')


@app.route('/signupsubmit', methods=['POST'])
def signupsubmit():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')


        existing_user = db.user_details.find_one({'email': email})
        if existing_user:
            return jsonify({'error': 'User already exists. Please sign in.'})


        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


        db.user_details.insert_one({'email': email, 'password': password_hash.decode('utf-8')})
        return jsonify({'success': 'User registered successfully.'})

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signinsubmit', methods=['POST'])
def signinsubmit():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = db.user_details.find_one({'email': email})
        if existing_user and bcrypt.checkpw(password.encode('utf-8'), existing_user['password'].encode('utf-8')):
            session['email'] = email
            print(session['email'])
            return jsonify({'success': 'Signed in successfully.'})
        else:
            return jsonify({'error': 'Invalid email or password. Please try again.'})


@app.route('/logout', methods=['GET','POST'])
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))


@app.route('/translation', methods=['GET', 'POST'])
def translation():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if not input_text:
            return jsonify({'error': 'No text provided for translation.'})

        import re


        def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
            if quantization == "4-bit":
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization == "8-bit":
                qconfig = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                )
            else:
                qconfig = None

            tokenizer = IndicTransTokenizer(direction=direction)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=qconfig,
            )

            if qconfig == None:
                model = model.to(DEVICE)
                if DEVICE == "cuda":
                    model.half()

            model.eval()

            return tokenizer, model

        def contains_explicit_words(sentence, bad_words_set):

            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if word in bad_words_set:
                    return True
            return False

        def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip, bad_words_set):
            translations = []
            for sentence in input_sentences:
                if contains_explicit_words(sentence, bad_words_set):
                    return ["Explicit or bad words detected. Translation aborted."]

            for i in range(0, len(input_sentences), BATCH_SIZE):
                batch = input_sentences[i: i + BATCH_SIZE]
                batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
                inputs = tokenizer(batch, src=True, truncation=True, padding="longest", return_tensors="pt",
                                   return_attention_mask=True).to(DEVICE)

                with torch.no_grad():
                    generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=1000,
                                                      num_beams=5, num_return_sequences=1)

                generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
                translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

                del inputs
                torch.cuda.empty_cache()

            return translations

        en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
        en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)

        ip = IndicProcessor(inference=True)

        en_sents = [input_text]

        src_lang, tgt_lang = "eng_Latn", "ben_Beng"

        explicit_words_set = {"shit", "damn", "bugger", "pissed", "bollocks"}
        hi_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip,
                                          explicit_words_set)
        translated_text = ""
        for input_sentence, translation in zip(en_sents, hi_translations):
            translated_text += translation + " "


        del en_indic_tokenizer, en_indic_model

        if 'email' in session:
            email=session["email"]
            db.translation_details.insert_one({'email': email, 'input-text': input_text, 'translated-text': translated_text})


        return jsonify({'translated_text': translated_text})
    else:
        return render_template('translation.html')

@app.route('/savepdf', methods=['GET','POST'])
def savepdf():

    screenshot = pyautogui.screenshot(region=(778, 335, 565, 330))

    screenshot_path = 'screenshot.png'
    screenshot.save(screenshot_path)

    with open(screenshot_path, "rb") as img_file:
        with open("output.pdf", "wb") as pdf_file:
            pdf_file.write(img2pdf.convert(img_file))

    pdf_path = 'output.pdf'
    response = make_response(send_file(pdf_path, as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename=translated-text.pdf"
    return response




if __name__ == '__main__':
    app.run()
