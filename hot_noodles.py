from lda import combine_text,compute_lda

#combine_text(csv_address="~/Downloads/userid_businessid_stars_text_100K.csv",new_csv_name="cleaned100K.csv")

out = compute_lda("cleaned100K.csv",25)

print(out)