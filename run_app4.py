import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pandas as pd
import tempfile
from scipy.signal import argrelextrema
from scipy.fftpack import fft, ifft
from moviepy.editor import *
from moviepy.editor import VideoFileClip

mp_drawing = mp.solutions.drawing_utils  # 検出した特徴点を描画
mp_drawing_styles = mp.solutions.drawing_styles  # 点や線の色、太さなどのスタイル
mp_pose = mp.solutions.pose  # 人間の身体の姿勢を推定g
st.title('ランニングフォーム診断AI')
st.write('あなたのフォームを診断します')
st.write('あなた1人を撮影したランニング動画をアップロードしてください')
st.write('動画サイズは200MB以下、形式はMP4, MOV, AVI, MPEG4のいずれかでお願いします')


def load_video(video_file):  # video_fileを読み込む関数
    tfile = tempfile.NamedTemporaryFile(delete=False) # tfileという一時ファイルを作成
    tfile.write(video_file.read())  #video_fileからデータを読み込み、tfileに格納

    # OpenCVで読み込むためにファイルパスを取得
    vf = cv2.VideoCapture(tfile.name)  # 上記で作成したtfileのパス=tfile.nameを取得し、ビデオファイルを開く
    return vf  

# Streamlitのファイルアップローダを使用してユーザーに動画ファイルのアップロードを依頼
uploaded_file = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
if uploaded_file is not None:
    # Save the uploaded video file to disk as we cannot manipulate it directly from the UploadedFile object
    with open('uploaded_video.mp4', 'wb') as out:
        out.write(uploaded_file.read())
    st.video('uploaded_video.mp4')

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)

    # Get the total duration of the video in seconds
    total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Allow user to select start and end times
    start_time = st.sidebar.slider("Start Time (seconds)", 0, total_duration, 0)
    end_time = st.sidebar.slider("End Time (seconds)", 0, total_duration, min(10, total_duration))
    end_time = min(end_time, start_time + 10)  # Ensure duration doesn't exceed 10 seconds

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

    # Create video writer with appropriate codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, size)

    # Read and write the selected frames
    for _ in range((end_time - start_time) * fps):
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    # Release resources
    cap.release()
    out.release()

    # Play the output video
    st.video('output.mp4')


video_file = 'output.mp4'



# def play_output_video(frames, fps):  # 与えたFrameとFPSからビデオを作成し再生する関数
#     # mp4形式の一時ファイルを作成
#     tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
#     temp_filename = tfile.name

#     # フレームサイズを取得
#     frame_height, frame_width, _ = frames[0].shape
#     size = (frame_width, frame_height)

#     # フレームを書き出す
#     out = cv2.VideoWriter(temp_filename, cv2.VideoWriter_fourcc(*'avc1'), fps, size)

#     for frame in frames:
#         # OpenCVはBGR形式を期待しているため、色の順番をRGBからBGRに変換
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(frame_bgr)
#     out.release()

#     # 作成した動画を再生
#     st.video(temp_filename, format='mp4')

if video_file is not None:
    # 動画ファイルをロード
    video_data = load_video(video_file)

    # 動画の情報を取得し、各フレームを読み込む
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))    
    video_hight = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_data.get(cv2.CAP_PROP_FPS))
    video_frames = video_data.get(cv2.CAP_PROP_FRAME_COUNT)
    st.write('FPS:', video_fps)
    st.write('Dimensions:', video_width, video_hight)
    video_data_array = []
    st.write('Video Frame', video_frames)

    #ファイルが正常に読み込めている間ループする
    while video_data.isOpened():
    #1フレームごとに読み込み
        success, image = video_data.read()
        if success:
            #フレームの画像を追加
            video_data_array.append(image)
        else:
            break
    video_data.release()
    read_frames = len(video_data_array)
    st.write('Frames Read:',int(read_frames))

    # 動画の全フレーム数を取得
    # total_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    # 読み込んだフレーム数を取得
    # read_frames = int(len(video_data_array))

    # 全フレーム数と読み込んだフレーム数を比較
    if video_frames != read_frames:
        st.write("ビデオの一部の読み取りに失敗しました。")
        st.write("ビデオファイルが壊れている、拡張子が違う、ビデオが重すぎるなどの原因が考えられます")
        st.write("読み取れた部分のみを使用し解析いたします。")
    else:
        st.write("正しく動画が読み込まれました")


    # 出力動画の設定
    # fourcc = cv2.VideoWriter_fourcc(*'avc1') # Be sure to use the lower case
    # out = cv2.VideoWriter('output.mp4', fourcc, video_fps, (video_width, video_hight))

    landmarks_list = []  # 各フレームで検出されたランドマークの座標を保存するリスト

    with mp_pose.Pose(
        static_image_mode=False,  # 連続したFrame間で姿勢推定を行う場合に、各Frameを独立した画像として扱うかどうか。
                                  # Falseとすれば、前のFrameの情報を利用してより正確な推定を行う。
        model_complexity=1, # 0,1,2のどれかを選択。2が最も精度が高いが計算コストも増える
        enable_segmentation=True,  # Trueとすることで、人間の部分と非人間の部分に分けて処理を行う
        min_detection_confidence=0.5) as pose:  # ランドマークの検出信頼度の最小値を設定。ここでは0.5未満のランドマークは無視される。

        #読み込んだ動画の各フレームに対してループを実行
        for loop_counter,image_data in enumerate(video_data_array):

            #姿勢推定モデルPOSEを使って、現在のフレーム画像を解析
            results = pose.process(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:  # ランドマークが検出されなかったフレームをスキップ
                continue

            # このフレームで検出されたランドマークのインデックスを表示
            detected_landmarks_indexes = [i for i, landmark in enumerate(results.pose_landmarks.landmark)]

            #検出されたランドマークを現在のフレーム画像に描画
            mp_drawing.draw_landmarks(
                image_data,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 描画されたフレームをリストに保存し直す
            video_data_array[loop_counter] = image_data

            # このフレームのランドマークの座標を取得し、リストに追加する
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:  # このFrameのランドマークに対してループを実行
                frame_landmarks.append([landmark.x, landmark.y])
            landmarks_list.append(frame_landmarks)

    # ランドマークの座標を含むリストをpandasのDataFrameに変換する
    landmarks_df = pd.DataFrame(landmarks_list)

    # DataFrameをCSVファイルとして保存する
    landmarks_df.to_csv('landmarks.csv', index=False)

    # 最初のフレームを表示
    # st.image(cv2.cvtColor(video_data_array[0], cv2.COLOR_BGR2RGB), caption="最初のフレーム", use_column_width=True)  # RGB形式の画像として表示

    # out.release()  # Be sure to release the video writer 

    # Open the video file
    # vf = cv2.VideoCapture('output.mp4')

    # Streamlitのvideoウィジェットで動画を再生する
    # st.video('output.mp4')

    # 動画出力処理
    # play_output_video(video_data_array, video_fps)

    import tempfile
    from moviepy.editor import *
    # 一時ファイルを作成
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    output_filename = tfile.name
    tfile2 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    output_filename2 = tfile2.name

    # 出力形式を指定する
    output_file = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'MP4V'),video_fps,(video_width,video_hight))

    # 動画出力処理
    for video_data in video_data_array:
        # OpenCVはBGR形式を期待しているため、色の順番をRGBからBGRに変換
        video_data_bgr = cv2.cvtColor(video_data, cv2.COLOR_RGB2BGR)
        output_file.write(video_data_bgr)
    output_file.release()

    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(output_filename)
    # RGBへの変換
    clip = clip.fl_image(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    clip.write_videofile(output_filename2, codec='libx264')

    # Streamlitのvideoウィジェットで動画を再生する
    st.video(output_filename2)




    # output_filename = 'output.mp4'
    # 出力形式を指定する
    # output_file = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'MP4V'),video_fps,(video_width,video_hight))
    #動画出力処理
    # for video_data in video_data_array:
        # output_file.write(video_data)
    # output_file.release()
    # Streamlitのvideoウィジェットで動画を再生する
    # st.video('output.mp4')

    # landmarks_df is your existing DataFrame
    num_landmarks = len(landmarks_df.columns) # assuming this is 33

    # Create new DataFrame to hold the separated x and y coordinates
    separated_df = pd.DataFrame()
    for i in range(num_landmarks):
        # Extract the x and y coordinates for each landmark
        separated_df[f'{i}_x'] = landmarks_df[i].apply(lambda coords: coords[0])
        separated_df[f'{i}_y'] = landmarks_df[i].apply(lambda coords: coords[1])

#ランナーの進行方向
    # ランダムに10個のフレームを選択
    random_frames = separated_df.sample(n=10)

    # 各フレームの結果を保存するリスト
    results = []

    # 各フレームで鼻のx座標と腰のx座標の平均を比較
    for i, frame in random_frames.iterrows():
        nose_x = frame['0_x']
        hip_x_avg = np.mean([frame['23_x'], frame['24_x']])
        
        if nose_x > hip_x_avg:
            results.append('right')
        else:
            results.append('left')

    # 結果の数を数える
    count_right = results.count('right')
    count_left = results.count('left')

    # 8つ以上のフレームで同じ結果が得られた場合にその結果を表示
    if count_right >= 8:
        st.write('このランナーは右方向に走っています。')
    elif count_left >= 8:
        st.write('このランナーは左方向に走っています。')
    else:
        st.write('このランナーの走る方向は一定ではありません。')



    for i in range(num_landmarks):
        # Extract the x and y coordinates for each landmark
        separated_df[f'{i}_x'] = landmarks_df[i].apply(lambda coords: coords[0])
        separated_df[f'{i}_y'] = landmarks_df[i].apply(lambda coords: coords[1])

    # Now separated_df should have 66 columns, with separated x and y coordinates for each landmark

    #print(separated_df.shape)

    # ランドマークのリストを作成します
    selected_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    # 選択したランドマークの 'x' と 'y' の列名を作成します
    selected_landmarks_x = [f"{i}_x" for i in selected_landmarks]
    selected_landmarks_y = [f"{i}_y" for i in selected_landmarks]

    # 新しい重心座標を計算します
    separated_df['centroid_x'] = separated_df[selected_landmarks_x].mean(axis=1)
    separated_df['centroid_y'] = separated_df[selected_landmarks_y].mean(axis=1)

    # separated_df is your DataFrame
    separated_df.to_csv('separated_landmarks.csv', index=False)

    # 重心の上下動可視化
    # fig, ax = plt.subplots()
    # ax.plot(separated_df['centroid_y'])
    # ax.set_title('Vertical Movement of Centroid')
    # ax.set_xlabel('Frame')
    # ax.set_ylabel('Position')
    # st.pyplot(fig)

    # 両足の上下動可視化
    fig, ax = plt.subplots()
    separated_df['left_foot_y_avg'] = separated_df[['27_y', '29_y']].mean(axis=1)
    separated_df['right_foot_y_avg'] = separated_df[['28_y', '30_y']].mean(axis=1)
    # ax.figure(figsize=(10,6))
    ax.plot(separated_df['left_foot_y_avg'], label='Left Foot')
    ax.plot(separated_df['right_foot_y_avg'], label='Right Foot')
    ax.set_title('Vertical Movement of Feet')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position')
    ax.legend()
    st.pyplot(fig)



    # Define the comparator for argrelextrema
    comp = np.greater

    # Convert the DataFrame column to a numpy array before FFT
    centroid_y = separated_df['centroid_y'].values

    # Get the FFT of the data
    fft_vals = fft(centroid_y)

    # Define the frequency threshold. We know that there are 30 frames per second, so if we assume that the runner cannot take more than 4 steps per second, this corresponds to a frequency of 4*30 = 120
    freq_threshold = 120

    # Zero out the FFT values above the threshold to remove high frequency components
    fft_vals_th = fft_vals.copy()
    fft_vals_th[freq_threshold:len(fft_vals)-freq_threshold] = 0

    # Get the filtered data by performing an inverse FFT
    filtered_data = ifft(fft_vals_th)

    # Get the real part of the filtered data
    filtered_data_real = filtered_data.real

    # Define the window size for the rolling average
    window_size = 5

    # Compute the rolling average
    smooth_data = pd.Series(filtered_data_real).rolling(window_size, center=True).mean()

    # Compute the local maxima of the smoothed data
    centroid_max_indices = argrelextrema(smooth_data.values, comp)
    valid_max_indices_centroid = centroid_max_indices[0]
    
    # Print the indices of the local maxima
    print(centroid_max_indices)
    print(f"Number of local maxima: {len(centroid_max_indices[0])}")

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(smooth_data, label='Smoothed centroid_y')
    plt.scatter(centroid_max_indices, smooth_data.iloc[centroid_max_indices], color='r') # Highlight local maxima
    plt.xlabel('Index')
    plt.ylabel('centroid_y')
    plt.title('Smoothed Centroid_Y values with local maxima highlighted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt)



    # Find the indices of relative minima for left and right foot
    left_heel_max_indices = argrelextrema(separated_df['29_y'].values, comp)
    left_toe_max_indices = argrelextrema(separated_df['31_y'].values, comp)

    right_heel_max_indices = argrelextrema(separated_df['30_y'].values, comp)
    right_toe_max_indices = argrelextrema(separated_df['32_y'].values, comp)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # 最も近いleft_heel_max_indicesを見つける
    nearest_left_heel_max_indices = [find_nearest(left_heel_max_indices[0], centroid_index) for centroid_index in valid_max_indices_centroid]
    left_heel_max_indices = nearest_left_heel_max_indices[::2]  # 奇数番目の要素のみを抽出

    nearest_right_heel_max_indices = [find_nearest(right_heel_max_indices[0], centroid_index) for centroid_index in valid_max_indices_centroid]
    right_heel_max_indices = nearest_right_heel_max_indices[1::2]  # 偶数番目の要素のみを抽出

    heel_max_indices = np.sort(np.concatenate((left_heel_max_indices, right_heel_max_indices)))

    # 簡単な比較のためにnumpy配列に変換
    valid_max_indices_centroid_array = np.array(valid_max_indices_centroid)

    # 配列を比較し、heel_max_indicesのフレーム値が小さいフレームの数をカウント
    heel_max_indices_count = np.sum(heel_max_indices < valid_max_indices_centroid_array)

    # 配列を比較し、centroid_max_indicesのフレーム値が小さいフレームの数をカウント
    centroid_smaller_count = np.sum(heel_max_indices > valid_max_indices_centroid_array)

    # フレーム数の取得
    total_frames = len(valid_max_indices_centroid)

    # 比率の計算
    ratio_heel_max_indices_smaller = heel_max_indices_count / total_frames
    ratio_centroid_smaller = centroid_smaller_count / total_frames
    ratio_equal = np.sum(heel_max_indices == valid_max_indices_centroid_array) / total_frames

    # st.write('重心よりも前に着地', round(ratio_heel_max_indices_smaller * 100, 2), '%')
    # st.write('真下着地', round((ratio_equal + ratio_centroid_smaller)*100, 2), '%')


#重心とかかとのx座標の比較
    # 初期化
    good_count = 0

    # 全てのフレームに対してループ
    for index in left_heel_max_indices + right_heel_max_indices:
        frame = separated_df.loc[index]
        
        # かかとのx座標と重心のx座標を取得します
        heel_x = frame['29_x'] if index in left_heel_max_indices else frame['30_x']  # 左かかとまたは右かかとのx座標
        center_x = frame['centroid_x']  # 重心のx座標
        
        # かかとのx座標と重心のx座標を比較します
        if count_right >= 8:
            if center_x > heel_x:
                good_count += 1
        else:
            if center_x < heel_x:
                good_count += 1

    # 全体のフレーム数に対するGOODのフレームの割合を計算します
    good_ratio = round(good_count / len(heel_max_indices) * 100, 2)

    st.write(f'あなたの歩数は {len(valid_max_indices_centroid)}歩です。')
    st.write(f'{len(valid_max_indices_centroid)}歩のうち、{good_ratio}％で真下着地ができています！')


    # 初期化
    left_good_count = 0
    right_good_count = 0

    # 左かかとのフレームに対してループ
    for index in left_heel_max_indices:
        frame = separated_df.loc[index]
        
        # かかとのx座標と重心のx座標を取得します
        heel_x = frame['29_x']  # 左かかとのx座標
        center_x = frame['centroid_x']  # 重心のx座標
        
        # かかとのx座標と重心のx座標を比較します
        if count_right >= 8:
            if center_x > heel_x:
                left_good_count += 1
        else:
            if center_x < heel_x:
                left_good_count += 1

    # 左かかとのGOODのフレームの割合を計算します
    left_good_ratio = round(left_good_count / len(left_heel_max_indices) * 100, 2)

    # 右かかとのフレームに対してループ
    for index in right_heel_max_indices:
        frame = separated_df.loc[index]
        
        # かかとのx座標と重心のx座標を取得します
        heel_x = frame['30_x']  # 右かかとのx座標
        center_x = frame['centroid_x']  # 重心のx座標
        
        # かかとのx座標と重心のx座標を比較します
        if count_right >= 8:
            if center_x > heel_x:
                right_good_count += 1
        else:
            if center_x < heel_x:
                right_good_count += 1

    # 右かかとのGOODのフレームの割合を計算します
    right_good_ratio = round(right_good_count / len(right_heel_max_indices) * 100, 2)

    st.write(f'左足は {left_good_ratio}％、右足は {right_good_ratio}％です！')

    # 左足と右足の比率を比較
    if left_good_ratio < right_good_ratio:
        difference = round(right_good_ratio - left_good_ratio, 2)  # 差を計算
        st.write(f'左足の方が右足より {difference}％ 多く重心より前に着地してしまっているようです。')
    elif right_good_ratio < left_good_ratio:
        difference = round(left_good_ratio - right_good_ratio, 2)  # 差を計算
        st.write(f'右足の方が左足より {difference}％ 多く重心より前に着地してしまっているようです。')
    else:
        st.write('左足と右足の比率は同じです。')
