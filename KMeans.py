# Muhammad Hasan Syadzily
# 1301194367

class KMeans:

  point = None
  inersia = None
  training = None
  
  def __init__(self, df: pd.DataFrame):
    # Kelas ini digunakan untuk menyiapkan dataframe yang akan ditraining. Pastikan kolom bernama id atau sejenis sudah di drop tidak termasuk ke dalam dataframe.
    print("K-Means akan ditentukan oleh atribut-atribut di bawah ini:")
    print("[", end="")
    for i in range(len(df.columns)):
      print(df.columns[i] + " ", end="")
    print("]", end="\n")
    self.training = df.to_numpy()
    
  def fit_predict(self, k_num:int = 3, max_step:int = 500, conv_threshold: float = 1e-5) -> np.array:
    # Membuat model KMeans dengan K tertentu. Akan mengkembalikan hasil prediksi cluster. Poin kluster akan disimpan pada variable point

    # Setting up cluster arry for every record
    cluster = np.zeros(len(self.training))
    
    # normalize data
    data = self.__normalize_data__(self.training)
    
    # Initialize centroids using kmeans++
    point = self.__initialize_centroids__(data, k_num)
        
    # Setup convergence and counter
    convergence = False
    step = 0 
        
    while not convergence and (step < max_step):
      initial_point = point
      distance = self.__calculate_distance__(data, point)
      cluster = self.__clustering__(distance)
      new_point = self.__point_nomralization__(data, point, cluster)
      convergence = self.__convergence_check__(initial_point, new_point, conv_threshold)
      if convergence:
        print("It's convergence!")
      else:
        point = new_point
        step += 1
        print("STEP:", step)
    self.inersia = self.__calculate_inersia__(data, cluster, point)
    self.point = self.__denormalize_point__(point, self.training)
    return cluster

  def get_cluster_centroid(self) -> np.array:
    # Fungsi ini digunakan untuk mengambil point
    if type(self.point) == "NoneType":
      print("Nothing returned, point not initialize. Try using fit_predict first.")
      return
    return self.point
  
  def __initialize_centroids__(self, data:np.array, k:np.array) -> np.array:
    # Fungsi ini digunakan untuk menginisialisasikan centroid. Menggunakan algoritma k-means++
    centroids = []
    centroids.append( data[random.randrange(0, len(data))] )
    
    for i in range(1, k):
      min_dist = []
      for data_point in data:
        distance_data_point = []
        for point in centroids:
          distance_data_point.append(np.linalg.norm(data_point - point))
        min_dist.append(min(distance_data_point))
      probcum  = sum(min_dist)
      prob_point = [value / probcum for value in min_dist]
      centroids.append(data[np.argmax(prob_point)])
    return np.array(centroids)
  
  def __clustering__(self, distance: np.array) -> np.array:
    # Fungsi ini akan mengembalikan hasil clustering berdasarkan distance
    cluster = np.zeros(len(distance))
    for i in range(len(cluster)):
      cluster[i] = np.argmin(distance[i])
    return cluster

