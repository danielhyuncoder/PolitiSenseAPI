def encrypt_tokenizer(tokenizer, file):
  file = open(file, "w")

  for item in tokenizer.word_index:
      file.write(str(item) + ":" + str(tokenizer.word_index[item]) + "\n")
