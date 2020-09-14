using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using System.Runtime.Serialization.Json;
using System.Net;

using Microsoft.Win32;
using System.Net.Http;

namespace MnistPredictor
{

    public class SendData
    {
        public string feature { get; set; }
    }

    public class ReadData
    {
        public string Success { get; set; }

        public string Prediction { get; set; }
    }

    /// <summary>
    /// MainWindow.xaml の相互作用ロジック
    /// </summary>
    public partial class MainWindow : Window
    {

        private string _SelectedImagePath = "";

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Img_button_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog();

            dialog.Filter = "画像ファイル(*.png)|*.png";

            if (dialog.ShowDialog() == true)
            {
                _SelectedImagePath = dialog.FileName;

                this.number_image.Source = new BitmapImage(new Uri(_SelectedImagePath));
            }
        }

        private string ConvertImageToBase64String(string imagePath)
        {
            FileStream fs = new FileStream(imagePath, FileMode.Open, FileAccess.Read);
            byte[] bs = new byte[fs.Length];

            int readBytes = fs.Read(bs, 0, (int)fs.Length);
            fs.Close();

            return Convert.ToBase64String(bs);
        }

        private void Pred_button_Click(object sender, RoutedEventArgs e)
        {

            if (_SelectedImagePath == "") {
                return;
            }

            var jsonString = "";
            var jsonData = new SendData()
            {
                feature = ConvertImageToBase64String(_SelectedImagePath)
            };


            using (var ms = new MemoryStream())
            using (var sr = new StreamReader(ms))
            {
                var serializer = new DataContractJsonSerializer(typeof(SendData));
                serializer.WriteObject(ms, jsonData);
                ms.Position = 0;
                jsonString = sr.ReadToEnd();
            }

            string url = "http://192.168.20.10:8081/predict";
            string webResponse = string.Empty;
            {
                Uri uri = new Uri(url);
                WebRequest httpWebRequest = (HttpWebRequest)WebRequest.Create(uri);
                httpWebRequest.ContentType = "application/json";
                httpWebRequest.Method = "POST";

                using (var sw = new StreamWriter(httpWebRequest.GetRequestStream()))
                {
                    sw.Write(jsonString);
                    sw.Flush();
                    sw.Close();

                    HttpWebResponse httpWebResponse = (HttpWebResponse)httpWebRequest.GetResponse();
                    using (StreamReader streamReader = new StreamReader(httpWebResponse.GetResponseStream()))
                    {
                        webResponse = streamReader.ReadToEnd();
                    }
                }
            }

            ReadData readData;
            using (var ms = new MemoryStream(Encoding.UTF8.GetBytes(webResponse))) {
                var sr = new DataContractJsonSerializer(typeof(ReadData));
                readData = (ReadData)sr.ReadObject(ms) as ReadData;
            }

            this.PredictResult.Text = readData.Prediction;
        }
    }
}
