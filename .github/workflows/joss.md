name: Draft PDF

on:
  push:
    paths:
      - joss/**
      - .github/workflows/draft-pdf.yml

jobs:
  paper:
    runs-on: ubuntu-latest
    name: Paper Draft
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      - name: Build draft PDF
        uses: openjournals/openjournals-draft-action@master
        with:
          journal: joss
          paper-path: joss/paper.md
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: paper
          path: joss/paper.pdf