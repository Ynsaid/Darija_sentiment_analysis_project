import wordcloud_generator as wordcloud


train_path = r"C:\Users\Younes\Desktop\M2\Nlp\Project\datasets\train.csv"
val_path   = r"C:\Users\Younes\Desktop\M2\Nlp\Project\datasets\validation.csv"
test_path  = r"C:\Users\Younes\Desktop\M2\Nlp\Project\datasets\test.csv"

df_raw = wordcloud.load_data(train_path, val_path, test_path)

wordcloud.visualize_class_distribution(df_raw, "Class Distribution", palette="coolwarm")

print("Generating 'Before' Word Cloud...")
text_raw = " ".join(df_raw['text'].astype(str).tolist())
wordcloud.generate_wordcloud(text_raw, "Word Cloud BEFORE Preprocessing")

print("Cleaning text and generating 'After' Word Cloud...")
text_clean = " ".join(df_raw['text'].astype(str).apply(wordcloud.clean_text).tolist())
wordcloud.generate_wordcloud(text_clean, "Word Cloud AFTER Preprocessing")

print("Analysis complete.")
