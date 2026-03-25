from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

with open("cleaned data/clean_corpus.txt","r",encoding="utf-8") as f:
    text = f.read()

wordcloud = WordCloud(width=800,height=400).generate(text)

plot_dir = "plot"
os.makedirs(plot_dir, exist_ok=True)
output_path = os.path.join(plot_dir, "wordcloud.png")

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Word cloud saved to: {output_path}")