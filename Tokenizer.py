class tokenizer:
  def decode_file(self, file_name):
    lines=open(file_name, "r").readlines()
    for line in lines:
      s=line.split(":")
      if len(s) < 2:
        continue
      self.word_index["".join(s[0].split(" "))]=int("".join("".join(s[1].split(" ")).split("\n")))
  def __init__(self, file_name):
    self.word_index={}
    self.decode_file(file_name)
  def text_to_sequence(self, text):
    words=text.split(" ")
    sequence = []
    for word in words:
      if word not in self.word_index:
        sequence.append(0)
      else:
        sequence.append(self.word_index[word])
    return sequence
