from PizzaManager import *

def addEditedFeature(df):
    df['post_was_edited'] = (df.request_text!=df.request_text_edit_aware)
    return df
