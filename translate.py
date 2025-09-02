from googletrans import Translator

translator = Translator()

def translate_text(text, dest='fr'):
    return translator.translate(text, dest=dest).text
