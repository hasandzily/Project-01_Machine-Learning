class KMeans:
  point = None
  inertia = None
  training_array = None
  
  # Fungsi dari kelas ini yaitu untuk mempersiapkan dataframe yang akan ditraining setelah dilakukannya data cleansing
  def __init__(self, df: pd.DataFrame):
    print("[", end="")
    for i in range(len(df.columns)):
      print(df.columns[i] + " ", end="")
    print("]", end="\n")
    self.training_array = df.to_numpy()
  
  # Fungsi ini berguna untuk membuat model K-Means dengan nilai K yang ditentukan dan menyimpan point cluster pada variabel point
  def prediksi_fit(self, k_num:int = 3, max_langkah:int = 500, conv_threshold: float = 1e-5) -> np.array:
    cluster = np.zeros(len(self.training_array))  # Mempersiapkan cluster array untuk setiap record
    data = self.__normalisasi_data__(self.training_array) # Untuk menormalisasi data
    point = self.__inisialisasi_centroids__(data, k_num) # Untuk menginisialisasikan centroids menggunakan KMeans++
    konvergen = False
    langkah = 0
    while not konvergen and (langkah < max_langkah):
      inisial_point = point
      distance = self.__hitung_distance__(data, point)
      cluster = self.__clustering__(distance)
      point_baru = self.__point_normalisasi__(data, point, cluster)
      konvergen = self.__konvergen_cek__(inisial_point, point_baru, conv_threshold)
      if konvergen:
        point = point_baru
        print("Adalah konvergen")
      else:
        point = point_baru
        langkah += 1
        print("Langkah:", langkah)
    self.inertia = self.__hitung_inertia__(data, cluster, point)
    self.point = self.__denormalisasi_point__(point, self.training_array)
    return cluster

  # Fungsi ini berguna untuk mengambil point
  def get_cluster_centroid(self) -> np.array:
    if type(self.point) == "NoneType":
      print("Point tidak terinisialisasi")
      return
    return self.point
  
  # Fungsi ini berguna untuk menginisialisasikan centroid dengan menggunakan KMeans++
  def __inisialisasi_centroids__(self, data:np.array, k:np.array) -> np.array:
    centroids = []
    centroids.append(data[random.randrange(0, len(data))])
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
  
  # Fungsi ini berguna untuk mengembalikan hasil clustering berdasarkan distance
  def __clustering__(self, distance: np.array) -> np.array:
    cluster = np.zeros(len(distance))
    for i in range(len(cluster)):
      cluster[i] = np.argmin(distance[i])
    return cluster

  # Fungsi ini berguna untuk menghitung setiap titik dengan point dan mengembalikan distance dari titik ke point
  def __hitung_distance__(self, data:np.array, point: np.array) -> np.array:
    distance = np.zeros((len(data), len(point)))
    for i in range(len(data)):
      current_record = data[i]
      for j in range(len(point)):
        current_point = point[j]
        distance[i][j] = np.linalg.norm(current_point - current_record) # Menggunakan numpy euclidean distance
    return distance
  
  # Fungsi ini berguna untuk menghitung ulang point dengan rata-rata
  def __point_normalisasi__(self, data:np.array, point:np.array, cluster:np.array) -> (np.array, np.array):
    point_baru = np.zeros((len(point), len(point[0])))
    counter_array = np.zeros(len(point))
    for i in range(len(cluster)):
      point_baru[int(cluster[i])] = point_baru[int(cluster[i])] + data[i]
      counter_array[int(cluster[i])] += 1
    unique_on_cluster = np.unique(cluster)
    for i in range(len(point)):
      # Mengatasi Atribut yang memiliki data "NaN"
      if i not in unique_on_cluster:
        point_baru[i] = point[i]
      else:
        point_baru[i] = np.true_divide(point_baru[i], counter_array[i])
    return point_baru

  # Fungsi ini berguna untuk mengecek konvergen berdasarkan threshold yang dibuat. Lalu titik cluster pertama akan dibandingkan dengan yang kedua.
  def __konvergen_cek__(self, points1: np.array, points2:np.array, threshold: float) -> bool:
    lokal_konvergen = False
    normalisasi_threshold_positive, normalisasi_threshold_negative  = 1 + threshold, 1 - threshold
    points_counter = 0
    center = np.zeros(len(points1[0]))
    for i in range(len(points1)):
      current_first_point, current_second_point = points1[i], points2[i]
      distance_first_point, distance_second_point = np.linalg.norm(current_first_point - center), np.linalg.norm(current_second_point - center)
      distance_threshold_positive = distance_first_point * normalisasi_threshold_positive
      distance_threshold_negative = distance_first_point * normalisasi_threshold_negative
      if distance_threshold_positive > distance_second_point and distance_threshold_negative < distance_second_point:
        points_counter += 1
    if points_counter == len(points1):
      lokal_konvergen = True
    return lokal_konvergen
  
  # Inertia mengukur seberapa baik dataset yang diklusterisasi menggunakan KMeans 
  # Dengan cara mengukur distance antara setiap data point dan centroidnya, mengkuadratkan distance, dan menjumlahkannya melewati satu cluster
  def __hitung_inertia__(self, data:np.array, cluster:np.array, points:np.array) -> np.array:
    inertia = 0
    for i in range(len(data)):
      inertia += (np.linalg.norm(data[i] - points[int(cluster[i])]))**2
    return inertia
  
  # Fungsi ini berguna untuk menormalisasikan data menggunakan min-max scaling 
  # Sampai data berjenis dan bersatuan dapat diproses dengan baik
  def __normalisasi_data__(self, data:np.array) -> np.array:
    data = data.copy()
    for i in range(len(data[0])):
      col_arr = data[:,i]
      minmax = MinMaxScaler()
      normalisasi = minmax.fit_transform(col_arr.reshape(-1,1)).reshape(1,-1)
      data[:, i] = normalisasi[0]
    return data
  
  # Fungsi ini berguna untuk mendenormalisasikan point-point yang sudah dihitung menggunakan data yang ternormalisasi
  def __denormalisasi_point__(self, data:np.array, original_data:np.array) -> np.array:
    for i in range(len(data[0])):
      col_arr = data[:,i]
      col_arr_ori = original_data[:, i]
      minimums = min(col_arr_ori)
      maximums = max(col_arr_ori)
      for j in range(len(col_arr)):
        col_arr[j] = ((col_arr[j] * (maximums - minimums)) + minimums)
      data[:, i] = col_arr
    return data