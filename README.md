# eye_gaze
Программа поддерживает три источника ввода данных:
1)	Статичное изображение
2)	Видео с веб-камеры 
3)	Видео из файла  

  
Процесс работы алгоритма: 
1.	Загрузка фотографии и изменение масштаба до 500x700 пикселей (такой масштаб необходим для адекватной работы функции определения расположения глаз)
2.	Выделение области глаз с помощью библиотеки opencv, затем полученное изображение обрезается на 20% сверху и снизу и переводится в оттенки серого. Для итогового изображения изменяю размер до 50x76 пикселей
3.	Перевожу данное изображение в векторную форму
4.	Предсказываю класс для этого вектора
Техническая справка:
Для решения данной задачи из готовых решений использовалась только функция opencv определения расположения глаз на фотографии.
Необходимо предустановить следующие библотеки:
•	cv2
•	numpy 
•	keras, tensorflow 
Необходимо поместить в папку со скриптом:
•	model_weights.h5 – веса нейронной сети
•	haarcascade_eye.xml 
В случае загрузки статичного изображения метка класса 1 отвечает за то, что человек смотрит в камеру, а 2 – не смотрит.
