import csv
import json
import requests

TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MWI1MGQ5YWYzMmY5ZTJhZDY2ZjBhZTAxOTkwNGM1NiIsInN1YiI6IjY1NTBkNzgwZDRmZTA0MDBmZTAzYzcxNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Mhoaiw48GVhQx2LJR8faT8Sj7fmcfq0wBYmz5PyXEpc'
CAST_COUNT = 5
KEYWORDS_COUNT = 5
CREW_COUNT = 5


def process_csv_file():
    combined_data = []
    with open('./movie/movie.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            movie_id = row[1]
            movie_data = fetch_movie_data(movie_id)
            if (movie_data):
                movie_data = {**{"movie_id": movie_id}, **movie_data}
                combined_data.append(movie_data)
            
    with open('./download/data.json', 'w') as json_file:
        json.dump(combined_data, json_file)   

def fetch(url):
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer " + TOKEN
    }

    response =  requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def fetch_movie_data(movie_id):
    print(movie_id);
    details = fetch(f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US")
    if (details == None):
        return dict()
    
    credits = fetch(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US")
    credits['cast'] = credits['cast'][:CAST_COUNT]
    credits['crew'] = credits['crew'][:CREW_COUNT]
    keywords = fetch(f"https://api.themoviedb.org/3/movie/{movie_id}/keywords")
    keywords['keywords'] = keywords['keywords'][:KEYWORDS_COUNT]
    combined = {**details, **credits, **keywords}
    return combined
    
    
    
process_csv_file()