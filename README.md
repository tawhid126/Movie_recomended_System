# 🎬 Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on genres, keywords, cast, and crew using machine learning.

## Features

- Content-based filtering using cosine similarity
- Natural Language Processing for text analysis
- Interactive Streamlit web interface
- Recommendations based on 5000+ movies from TMDB dataset

## Installation

1. Clone or navigate to the project directory:
```bash
cd /Users/tawhidurrahman/ML_projects/Movie_recommended_system
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install pandas numpy scikit-learn nltk streamlit
```

## Usage

### Generate Model Files

First, run the data processing script to generate the recommendation model:

```bash
python Recomended_syste.py
```

This will create:
- `movies.pkl` - Processed movie data
- `similarity.pkl` - Cosine similarity matrix

### Run the Web App

Launch the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` (or another port if 8501 is busy).

## How It Works

1. **Data Processing**: 
   - Loads movie data from TMDB dataset
   - Extracts features: genres, keywords, cast (top 3), and director
   - Applies stemming to normalize text
   - Creates feature vectors using CountVectorizer

2. **Similarity Calculation**:
   - Computes cosine similarity between all movies
   - Stores similarity matrix for fast recommendations

3. **Recommendations**:
   - Finds the selected movie's index
   - Retrieves top 5 most similar movies
   - Displays results in the web interface

## Dataset

The system uses the TMDB 5000 Movie Dataset:
- `tmdb_5000_movies.csv` - Movie metadata
- `tmdb_5000_credits.csv` - Cast and crew information

## Technologies

- **Python 3.13**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning (CountVectorizer, cosine similarity)
- **NLTK** - Natural language processing
- **Streamlit** - Web interface

## Project Structure

```
Movie_recommended_system/
├── Movie_database/
│   ├── tmdb_5000_credits.csv
│   └── tmdb_5000_movies.csv
├── Recomended_syste.py    # Data processing script
├── app.py                  # Streamlit web app
├── movies.pkl             # Processed movie data
├── similarity.pkl         # Similarity matrix
└── README.md
```

## License

This project is for educational purposes.
