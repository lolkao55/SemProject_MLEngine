import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title='NutriStrat AI Demo', layout='centered')
st.title(' NutriStrat — SVM Meal Nutrient & Recommender System')

# Load saved SVM artifacts
vectorizer = joblib.load('nutristrat_tfidf.joblib')
svm_model = joblib.load('nutristrat_svm.joblib')
enriched_df = pd.read_csv('enriched_food_data.csv')

TARGET_COLS = ['Protein', 'Carbs', 'Fats', 'Fiber_Vitamins']
THRESHOLD = 0.75

NUTRIENT_KEYWORDS = {
    'Protein': [
        'chicken',
        'beef',
        'fish',
        'tilapia',
        'omena',
        'egg',
        'lentil',
        'ndengu',
        'bean',
        'tofu',
        'paneer',
    ],
    'Carbs': [
        'rice',
        'ugali',
        'chapati',
        'mandazi',
        'githeri',
        'potato',
        'pasta',
        'bread',
        'flour',
    ],
    'Fats': [
        'oil',
        'butter',
        'ghee',
        'avocado',
        'almond',
        'peanut',
        'coconut',
        'nut',
    ],
    'Fiber_Vitamins': [
        'carrot',
        'spinach',
        'sukuma',
        'managu',
        'tomato',
        'onion',
        'kachumbari',
        'cabbage',
        'vegetable',
    ],
}

tab1, tab2 = st.tabs(['🔍 Live SVM Meal Diagnosis', '📊 Model Benchmark'])

with tab1:
  meal_input = st.text_input(
      'Enter your meal or ingredients:', value='Ugali and tilapia'
  )
  diet_filter = st.selectbox('Dietary Preference:', ['All', 'veg', 'non-veg'])

  if st.button('Analyze Meal with SVM', type='primary'):
    parts = [
        p.strip()
        for p in re.split(
            r',|\band\b|\bwith\b|\bplus\b', meal_input.lower()
        )
        if p.strip()
    ]
    vecs = vectorizer.transform([meal_input.lower()] + parts)
    probs = svm_model.predict_proba(vecs).max(axis=0)

    present = [n for n, p in zip(TARGET_COLS, probs) if p >= THRESHOLD]
    missing = [n for n, p in zip(TARGET_COLS, probs) if p < THRESHOLD]

    col1, col2 = st.columns(2)
    col1.success(
        f"**Present Nutrients:** {', '.join(present) if present else 'None'}"
    )
    if missing:
      col2.warning(f"**Missing Nutrients:** {', '.join(missing)}")
    else:
      col2.success('**Status:** Completely Balanced Meal!')

    # Recommend complementary dishes (excluding Desserts)
    if missing:
      candidates = enriched_df[
          enriched_df['C_Type'].str.lower() != 'dessert'
      ].copy()
      if diet_filter != 'All':
        candidates = candidates[
            candidates['Veg_Non'].str.lower() == diet_filter
        ]

      candidates['Missing_Covered'] = candidates[missing].sum(axis=1)
      target_query = ' '.join([' '.join(NUTRIENT_KEYWORDS[m]) for m in missing])
      sim_vec = TfidfVectorizer(stop_words='english')
      tfidf_mat = sim_vec.fit_transform(candidates['Describe'].fillna(''))
      candidates['Similarity'] = cosine_similarity(
          sim_vec.transform([target_query]), tfidf_mat
      ).flatten()
      candidates['Balance_Score'] = (
          candidates['Missing_Covered']
          + (candidates['Avg_Rating'] / 10.0) * 0.5
          + candidates['Similarity']
      )

      top_sides = (
          candidates[candidates['Missing_Covered'] > 0]
          .sort_values(by='Balance_Score', ascending=False)
          .head(3)
      )
      st.subheader('Top Recommended Complementary Dishes:')
      st.dataframe(
          top_sides[['Name', 'C_Type', 'Veg_Non', 'Avg_Rating'] + missing],
          hide_index=True,
      )

with tab2:
  st.subheader('80/20 Test Set Evaluation on Food Dataset')
  st.table(
      pd.DataFrame([
          {
              'Model': 'Decision Tree',
              'Micro-F1': 0.9502,
              'Macro-F1': 0.9497,
              'Exact Match Acc': '76.25%',
          },
          {
              'Model': 'SVM (RBF Kernel)',
              'Micro-F1': 0.9198,
              'Macro-F1': 0.9180,
              'Exact Match Acc': '56.25%',
          },
          {
              'Model': 'Naive Bayes (Multinomial)',
              'Micro-F1': 0.9072,
              'Macro-F1': 0.9055,
              'Exact Match Acc': '55.00%',
          },
      ])
  )