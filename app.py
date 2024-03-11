import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
import pandas as pd
from regex import escape
from urllib.request import urlopen
from io import BytesIO
# class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader
# class for loading datasets
from surprise.dataset import Dataset
# for implementing similarity based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

IMGSIZEW = 300
IMGSIZEH = 400

links = pd.read_csv('links.csv', dtype={"imdbId":"string","tmdbId":"string"})
meta = pd.read_csv('movies_metadata.csv')
creditsData = pd.read_csv('credits.csv')
creditsData = pd.DataFrame(creditsData, dtype='object')
rating = pd.read_csv('ratings.csv')
rating = rating.drop(['timestamp'], axis=1)
meta = meta[meta.adult == "False"]
result = None

UID = rating['userId'].iloc[-1] + 1

def search():
    # delete old results
    resultTree.delete(*resultTree.get_children())
    
    # if not searchEntry.get():
    #     return
    
    reg = r"(?i)[a-b]*" + escape(searchEntry.get()) + r"[a-b]*"
    global result
    result = meta[["title","release_date", "revenue", "id", "overview", "poster_path"]].where(meta["title"].str.contains(reg)).dropna().sort_values(by=["revenue"], ascending=False)
    resultList = result.to_numpy().tolist()
    for res in resultList:
        resultTree.insert("", tk.END, iid=res[3], values=[res[0],res[1], "${:,.0f}".format(res[2])])

def searchReturn(Event):
    search()

def treeview_sort_column(tv : ttk.Treeview, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    l.sort(reverse=reverse)

    # rearrange items in sorted positions
    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

    # reverse sort next time
    tv.heading(col, command=lambda _col=col: treeview_sort_column(tv, _col, not reverse))

def clearLabel():
    detailsLabel["image"]=""
    detailsLabel["text"]=""

def switch_button_on(event):
    for b in MovieButtons:
        b["state"] = "normal"

def switch_button_off(event):
    for b in MovieButtons:
        b["state"] = "disabled"

def switch_button(b, on: bool):
    if on:
        b["state"] = "disabled"
    else:
        b["state"] = "normal"

def cmd_rate(event):
    selectedID = resultTree.focus() # id in meta file
    if not selectedID:
        switch_button(b_rate)
        return
    
    score = b_rate.get()
    name = resultTree.item(selectedID)['values'][0]
    
    ratingID = set_movie(selectedID, meta, links) # id in ratings file
    ratedList.insert(0, f"{score} -> {name}")
    rating.loc[len(rating.index)] = [UID, ratingID, score]    

def cmd_poster():
    clearLabel()
    # setting image with label
    url = "https://image.tmdb.org/t/p/original"
    selectedID = resultTree.focus()
    
    if not selectedID:
        switch_button(b_desc)
        return
    
    data = pd.DataFrame(result)
    
    path = data["poster_path"].where(data["id"].str.contains(selectedID)).dropna().iloc[0]
    url+=path
    try:
        u = urlopen(url)
        raw_data = u.read()
        u.close()
    
        im = Image.open(BytesIO(raw_data))

        im = im.resize((IMGSIZEW,IMGSIZEH))
        photo = ImageTk.PhotoImage(im)
        detailsLabel.config(image=photo)
        detailsLabel.image = photo

    except Exception:
        movName = resultTree.item(selectedID)['values'][0]
        detailsLabel["text"] = f"Sorry :/ image for {movName} was not found."

def cmd_desc():
    clearLabel()
    selectedID = resultTree.focus()
    
    if not selectedID:
        switch_button(b_desc)
        return
    
    desc = pd.DataFrame(result)
    target = desc["overview"].where(desc["id"].str.contains(selectedID)).dropna().iloc[0]
    detailsLabel["text"] = target

def cmd_cast():
    clearLabel()
    selectedID = resultTree.focus()
    
    if not selectedID:
        switch_button(b_cast)
        return

    target = creditsData[creditsData["id"] == int(selectedID)]["cast"].iloc[0]
    target = eval(target)
    tList = []
    for t, i in zip(target, range(15)):
        tList.append(f'{t["character"]}: {t["name"]}\n')
    detailsLabel["text"] = ''.join(tList)

def top_rated_movies(data, n, min_interaction=100):
    #Calculating average ratings
    average_rating = data.groupby('movieId').mean()['rating']
    #Calculating the count of ratings
    count_rating = data.groupby('movieId').count()['rating']
    #Making a dataframe with the count and average of ratings
    final_data = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})
    #Finding movies with minimum number of interactions
    recommendations = final_data[final_data['rating_count'] > min_interaction]
    #Sorting values w.r.t average rating
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    
    return recommendations.index[:n]

def get_recommendations(data, user_id, top_n, algo):
    
    # creating an empty list to store the recommended movie ids
    recommendations = []
    
    # creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    
    # extracting those movie ids which the user_id has not interacted yet
    non_interacted_movies = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # looping through each of the movie id which user_id has not interacted yet
    for item_id in non_interacted_movies:
        
        # predicting the ratings for those non interacted movie ids by this user
        est = algo.predict(user_id, item_id).est
        
        # appending the predicted ratings
        recommendations.append((item_id, est))

    # sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_n]

def get_movie(movieId, meta, links):
    imdbID = 'tt' + str(links[links['movieId'] == movieId]['imdbId'].values[0])
    return meta[meta['imdb_id'] == imdbID]

def set_movie(idmov, meta, links):
    imdb = meta[meta['id'] == idmov]['imdb_id'].values[0]
    return links[links['imdbId'] == imdb[2:]]['movieId'].values[0]

def recommend():
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5), skip_lines=1)
    # loading the rating dataset
    data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
    # splitting the data into train and test dataset
    trainset = data.build_full_trainset()
    # optimized hyperparamterization
    similarity_algo_optimized_user = KNNBasic(sim_options={'name': 'cosine', 'user_based': True, "min_support":2}, k=30, min_k=3, random_state=1, verbose=False)
    similarity_algo_optimized_user.fit(trainset)
    
    try:
        recommendations = get_recommendations(rating, UID, 20, similarity_algo_optimized_user)
    except:
        showerror("Error", "Error getting recommendations, try restarting the app.")
        return
    
    return recommendations

def cmd_recommend():
    global result
    clearLabel()
    resultTree.delete(*resultTree.get_children())
    recommendations = recommend()
    movList = []
    for rec in recommendations:
        movID = rec[0]
        try:
            movie = get_movie(movID,meta,links)
            movList.append(movie[["title","release_date", "revenue", "id", "overview", "poster_path"]].values.tolist()[0])
        except:
            # print("id not found.")
            continue
        
            
        # showerror("Error", "Error getting recommendations, try restarting the app..")
        # return
            
    result = pd.DataFrame(movList, columns=["title","release_date", "revenue", "id", "overview", "poster_path"])
    for res in movList:
        resultTree.insert("", tk.END, iid=res[3], values=res[:3])

if __name__ == "__main__":
    # creating main tkinter window/toplevel
    root = tk.Tk()
    root.title('The Recommendinator!')
    root.resizable(0,0)

    master = tk.Frame(root)

    # entry widgets, used to take entry from user
    searchEntry = tk.Entry(master, width=40)
    searchEntry.grid(row=0, column=0, columnspan=3, padx=4, sticky=tk.N, pady=2)
    searchEntry.bind("<Return>", searchReturn)
    searchButton = tk.Button(master, text="Search", command=search)
    searchButton.grid(row=0, column=3, sticky=tk.N, pady=2)

    # treeview result
    resultTree = ttk.Treeview(master)
    column_list_account = ["Title", "Release date", "Revenue"]
    resultTree["columns"] = column_list_account
    resultTree["show"] = "headings"  # removes empty column

        
    for column in column_list_account:
        resultTree.heading(column, text=column, command=lambda _col=column: treeview_sort_column(resultTree, _col, False))

    resultTree.column("Title", width=200)
    resultTree.column("Release date", width=150)
    resultTree.column("Revenue", width=100)
        
    resultTree.grid(row=1, column=0, columnspan=4, sticky=tk.NW, padx=6, pady=(0,5))

    treescroll = tk.Scrollbar(master)
    treescroll.configure(command=resultTree.yview)
    resultTree.configure(yscrollcommand=treescroll.set)

    ratedList = tk.Listbox(master, width=30)
    ratedList.grid(row=2, column=0, columnspan=4, sticky=tk.N, padx=6)

    detailsLabel = tk.Label(master, wraplength=IMGSIZEW, justify=tk.LEFT)
    detailsLabel.grid(row=0, column=5, columnspan=4, rowspan=3, padx=5, pady=(3,0))

    # button widget
    b_rate = ttk.Combobox(master, state="disable", values=("5","4.5","4","3.5","3","2.5","2","1.5","1"), width=10)
    b_poster = tk.Button(master, text="Poster", state="disable", command=cmd_poster)    
    b_desc = tk.Button(master, text="Description", state="disable", command=cmd_desc)
    b_cast = tk.Button(master, text="Cast", state="disable", command=cmd_cast)
    b_recommend = tk.Button(master, text="RECOMMEND!", command=cmd_recommend)

    # arranging button widgets
    b_poster.grid(row=3, column=5, sticky=tk.W, pady=(5,10))
    b_desc.grid(row=3, column=6, sticky=tk.E, pady=(5,10))
    b_cast.grid(row=3, column=7, sticky=tk.E, pady=(5,10))
    b_rate.grid(row=3, column=8, sticky=tk.E, pady=10)
    b_recommend.grid(row=3, column=0, sticky=tk.W, pady=(5,10))
    
    MovieButtons = [
        b_poster,
        b_desc,
        b_cast,
        b_rate
        ]

    resultTree.bind("<Button-1>", switch_button_on)
    b_rate.bind("<<ComboboxSelected>>", cmd_rate)

    # master.place(relx=0.5,rely=0.5,anchor=tk.CENTER)
    # master.grid(sticky=tk.NSEW)

    master.pack()
    root.mainloop()