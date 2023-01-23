all: slides.pdf

slides.pdf: slides.tex dfm.pdf banana.png corner.png
	pdflatex -shell-escape slides
	pdflatex -shell-escape slides

# -shell-escape is for the "minted" package which does source-code highlighting

dfm.pdf:
	wget https://files.speakerdeck.com/presentations/1cbbeda07de1013062221231381d8c65/vanderbilt.pdf -O $@
