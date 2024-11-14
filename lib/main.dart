import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:audioplayers/audioplayers.dart';

void main() async {
  // Flutter 엔진 초기화
  WidgetsFlutterBinding.ensureInitialized();
  // 사용 가능한 카메라 목록 가져오기
  final cameras = await availableCameras();
  // 앱 실행 카메라 정보를 전달
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '졸음 감지 시스템',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark, // 다크 테마 적용 (카메라 프리뷰에 더 적합)
      ),
      home: DrowsinessDetection(cameras: cameras),
    );
  }
}

class DrowsinessDetection extends StatefulWidget {
  final List<CameraDescription> cameras;

  const DrowsinessDetection({
    super.key,
    required this.cameras,
  });

  @override
  DrowsinessDetectionState createState() => DrowsinessDetectionState();
}

class ModelHelper {
  static Future<void> _normalizeImage(List<List<List<double>>> buffer) async {
    double imageMean = 127.5;
    double imageStd = 127.5;

    for (int y = 0; y < buffer.length; y++) {
      for (int x = 0; x < buffer[y].length; x++) {
        for (int c = 0; c < buffer[y][x].length; c++) {
          buffer[y][x][c] = (buffer[y][x][c] - imageMean) / imageStd;
        }
      }
    }
  }
}

class DrowsinessDetectionState extends State<DrowsinessDetection>
    with WidgetsBindingObserver {
  late CameraController _controller;
  final audioPlayer = AudioPlayer();
  Interpreter? _interpreter;
  DateTime? lastProcessedTime;

  // 성능 최적화를 위한 상수들
  static const Duration processInterval =
      Duration(milliseconds: 100); // 디바이스 성능에 따라 적절히 조절
  static const int modelInputSize = 160; // 모델 입력 크기

  // 이미지 처리를 위한 버퍼
  late List<List<List<double>>> _inputBuffer;
  // 모델 출력 버퍼 (softmax 출력: [눈감음확률, 눈뜸확률])
  final List<double> _outputBuffer = List.filled(2, 0.0);

  // 졸음 감지를 위한 상태 변수들
  int drowsyFrameCount = 0;
  static const int drowsyFrameThreshold = 10;
  DateTime? lastAlertTime;
  bool isProcessing = false; // 프레임 처리 중복 방지

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); // 앱 생명주기 관찰자 등록
    _initializeCamera(); // 카메라 초기화
    _loadModel(); // AI 모델 로드
    _initializeBuffers(); // 입력 버퍼 초기화
  }

  void _initializeBuffers() {
    _inputBuffer = List.generate(
      modelInputSize,
      (_) => List.generate(
        modelInputSize,
        (_) => List.filled(3, 0.0),
        growable: false,
      ),
      growable: false,
    );
  }

  // 카메라 초기화 및 스트림 시작
  Future<void> _initializeCamera() async {
    try {
      _controller = CameraController(
        widget.cameras[1], // 전면 카메라 선택 (index 1)
        ResolutionPreset.low, // 저해상도
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
        fps: 20, // 초당 프레임 수 제한
      );
      await _controller.initialize();

      // 무한 루프. 카메라로 부터 실시간 이미지 스트림 받아옴
      await _controller.startImageStream((CameraImage image) {
        final now = DateTime.now();
        // 프레임 처리 간격 및 중복 처리 체크
        if (!isProcessing && lastProcessedTime == null ||
            now.difference(lastProcessedTime!) >= processInterval) {
          isProcessing = true;
          lastProcessedTime = now;
          _processCameraImage(image).then((_) {
            // 이미지 처리 시작
            isProcessing = false;
          });
        }
      });
      setState(() {});
    } on CameraException catch (e) {
      print('카메라 초기화 오류: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      final options = InterpreterOptions()..threads = 4; // 멀티스레딩 활성화
      //..useNnapi = true; // Android Neural Networks API 사용  텐서플로우2.2이상부터 우린 2.14...

      _interpreter = await Interpreter.fromAsset(
        'assets/train_jt50_merged_mov2.tflite',
        //'assets/train_jt_merged_mov2.tflite', //  그나마 조음
        // 'assets/train_merged_mov2.tflite',//  별로
        // 'assets/train_merged_cnn_light.tflite',// 구림
        options: options,
      );

      print('Input Tensor Shape: ${_interpreter!.getInputTensor(0).shape}');
      print('Output Tensor Shape: ${_interpreter!.getOutputTensor(0).shape}');
      print('Input Tensor Type: ${_interpreter!.getInputTensor(0).type}');
      print('Output Tensor Type: ${_interpreter!.getOutputTensor(0).type}');
      // 입력 데이터 형식 확인
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      print('Input Tensor Details: ${inputTensor.toString()}');
      print('Output Tensor Details: ${outputTensor.toString()}');
    } catch (e) {
      // ignore: avoid_print
      print('모델 로드 오류: $e');
      print(e.toString());
    }
  }

  // 카메라 이미지 처리
  Future<void> _processCameraImage(CameraImage image) async {
    if (_interpreter == null) return;

    try {
      // 이미지 전처리
      final int width = image.width;
      final int height = image.height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel!;

      // YUV to RGB 변환을 위한 임시 버퍼
      List<List<List<int>>> rgbBuffer = List.generate(
        modelInputSize,
        (_) => List.generate(
          modelInputSize,
          (_) => List.filled(3, 0),
          growable: false,
        ),
        growable: false,
      );

      // YUV420 to RGB 변환 및 크기 조정
      for (int x = 0; x < modelInputSize; x++) {
        for (int y = 0; y < modelInputSize; y++) {
          // int sourceX = (x * width ~/ modelInputSize);
          // int sourceY = (y * height ~/ modelInputSize);
          // 이미지 뒤집기 (전면 카메라 미러링 처리)
          int sourceX = width - 1 - (x * width ~/ modelInputSize);
          int sourceY = (y * height ~/ modelInputSize);
          // // 이미지 중앙 부분에 집중
          // int sourceX = (x * width ~/ modelInputSize);
          // int sourceY = (y * height ~/ modelInputSize);

          final int uvIndex =
              uvPixelStride * (sourceX ~/ 2) + uvRowStride * (sourceY ~/ 2);
          final int index = sourceY * width + sourceX;

          // // YUV 값 추출
          // final yp = image.planes[0].bytes[index];
          // final up = image.planes[1].bytes[uvIndex];
          // final vp = image.planes[2].bytes[uvIndex];

          // YUV 값 추출
          int yp = image.planes[0].bytes[index] & 0xFF;
          int up = image.planes[1].bytes[uvIndex] & 0xFF;
          int vp = image.planes[2].bytes[uvIndex] & 0xFF;

          // BT.601 변환 공식 적용
          int y1 = yp;
          int u = up - 128;
          int v = vp - 128;

          // RGB 변환 (정수 연산으로 최적화)
          int r = y1 + ((1402 * v) >> 10);
          int g = y1 - ((344 * u + 714 * v) >> 10);
          int b = y1 + ((1772 * u) >> 10);

          // RGB 값 클램핑
          rgbBuffer[y][x][0] = r.clamp(0, 255);
          rgbBuffer[y][x][1] = g.clamp(0, 255);
          rgbBuffer[y][x][2] = b.clamp(0, 255);

          // // YUV to RGB 변환
          // int r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
          // int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
          //     .round()
          //     .clamp(0, 255);
          // int b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

          // // 이미지 정규화 (-1 ~ 1 범위로) (모델 전처리에 맞춤):
          // _inputBuffer[y][x][0] = (r / 127.5) - 1;
          // _inputBuffer[y][x][1] = (g / 127.5) - 1;
          // _inputBuffer[y][x][2] = (b / 127.5) - 1;
        }
      }

      // 정규화 전 RGB 통계
      List<double> means = List.filled(3, 0.0);
      List<double> maxVals = List.filled(3, 0.0);
      List<double> minVals = List.filled(3, 255.0);

      for (int y = 0; y < modelInputSize; y++) {
        for (int x = 0; x < modelInputSize; x++) {
          for (int c = 0; c < 3; c++) {
            means[c] += rgbBuffer[y][x][c];
            maxVals[c] = max(maxVals[c], rgbBuffer[y][x][c].toDouble());
            minVals[c] = min(minVals[c], rgbBuffer[y][x][c].toDouble());
          }
        }
      }

      for (int c = 0; c < 3; c++) {
        means[c] /= (modelInputSize * modelInputSize);
      }

      print('RGB Statistics:');
      for (int c = 0; c < 3; c++) {
        print(
            'Channel $c - Min: ${minVals[c]}, Max: ${maxVals[c]}, Mean: ${means[c]}');
      }

      // MobileNetV2 정규화 적용 (1/127.5 - 1)
      for (int y = 0; y < modelInputSize; y++) {
        for (int x = 0; x < modelInputSize; x++) {
          for (int c = 0; c < 3; c++) {
            _inputBuffer[y][x][c] = (rgbBuffer[y][x][c] / 127.5) - 1.0;
          }
        }
      }

      // 모델 실행 전 입력 데이터 확인
      double inputMin = double.infinity;
      double inputMax = double.negativeInfinity;
      double inputSum = 0;
      int count = 0;

      for (var row in _inputBuffer) {
        for (var pixel in row) {
          for (var value in pixel) {
            inputMin = min(inputMin, value);
            inputMax = max(inputMax, value);
            inputSum += value;
            count++;
          }
        }
      }

      print('Normalized input statistics:');
      print('Min: $inputMin');
      print('Max: $inputMax');
      print('Mean: ${inputSum / count}');

      final input = [_inputBuffer];
      final output = [_outputBuffer];

      // 모델 실행
      _interpreter!.run(input, output);

      // 원본 출력값 확인
      print('Raw outputs: ${output[0].toString()}');

      // 소프트맥스 적용
      double maxOutput = output[0].reduce(max);
      List<double> expOutputs =
          output[0].map((x) => exp(x - maxOutput)).toList();
      double sumExp = expOutputs.reduce((a, b) => a + b);
      for (int i = 0; i < output[0].length; i++) {
        output[0][i] = expOutputs[i] / sumExp;
      }

      print('Softmax outputs: ${output[0].toString()}');

      // 결과 처리 (label.txt = [close, open]) (눈 감음 확률이 더 높은지 확인)
      //final isEyeClosed = output[0][0] > output[0][1];
      // 임계값 적용
      final isEyeClosed = output[0][0] > 0.55; // 임계값 조정

      if (mounted) {
        setState(() {
          if (isEyeClosed) {
            drowsyFrameCount++;
            print('Drowsy frame count: $drowsyFrameCount');
            if (drowsyFrameCount >= drowsyFrameThreshold) {
              _handleDrowsiness();
            }
          } else {
            //drowsyFrameCount = 0;
            drowsyFrameCount = max(0, drowsyFrameCount - 2); // 점진적 감소
            //졸음 상태가 순간적으로 해제되었다가 다시 발생할 때 더 빠르게 감지
          }
        });
      }
    } catch (e) {
      print('Error processing frame: $e');
      print(e.toString());
    }
  }

  // 졸음 감지 시 처리
  void _handleDrowsiness() {
    final now = DateTime.now();
    if (lastAlertTime == null ||
        now.difference(lastAlertTime!) > const Duration(seconds: 5)) {
      _showAlert();
      lastAlertTime = now;
      drowsyFrameCount = 0;
      //drowsyFrameCount = drowsyFrameThreshold ~/ 2; // 절반으로 리셋
      //완전히 리셋(0)하면 다음 알림까지 너무 오래 걸릴 수 있음
      //알림 후에도 여전히 졸린 상태라면 더 빠르게 다음 알림 발생
    }
  }

  // 경고 표시 및 소리 재생
  void _showAlert() {
    audioPlayer.play(AssetSource('alert.wav'));

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Row(
            children: [
              Icon(Icons.warning, color: Colors.white),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  '졸음이 감지되었습니다! 휴식이 필요합니다.',
                  style: TextStyle(fontSize: 16),
                ),
              ),
            ],
          ),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 3),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('실시간 졸음 감지'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info),
            onPressed: () => _showStatusDialog(),
            tooltip: '상태 정보',
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_controller), // 카메라 프리뷰를 전체 화면으로 표시
          Positioned(
            // 상태 정보 오버레이
            top: 10,
            right: 10,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(8),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 4,
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '눈 감은 확률: ${(_outputBuffer[0] * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: _getWarningColor(_outputBuffer[0]),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '연속 프레임: $drowsyFrameCount',
                    style: TextStyle(
                      color: _getWarningColor(
                          drowsyFrameCount / drowsyFrameThreshold),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // 경고 색상 계산
  Color _getWarningColor(double value) {
    if (value < 0.3) return Colors.green;
    if (value < 0.7) return Colors.yellow;
    return Colors.red;
  }

  // 상태 정보 다이얼로그
  void _showStatusDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('상태 정보'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('눈 감은 상태 확률: ${(_outputBuffer[0] * 100).toStringAsFixed(1)}%'),
            Text('눈 뜬 상태 확률: ${(_outputBuffer[1] * 100).toStringAsFixed(1)}%'),
            Text('연속 프레임: $drowsyFrameCount'),
            const Text('알림 간격: 5초'),
            const Divider(),
            Text('마지막 알림: ${lastAlertTime?.toString() ?? "없음"}'),
            const SizedBox(height: 8),
            const Text('* 프레임 처리 간격: 100ms\n* 졸음 감지 임계값: 10프레임',
                style: TextStyle(fontSize: 12, color: Colors.grey)),
          ],
        ),
        actions: [
          TextButton(
            child: const Text('확인'),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // 앱 생명주기 변화 처리
    if (state == AppLifecycleState.inactive) {
      _controller.stopImageStream();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller.dispose();
    _interpreter?.close();
    audioPlayer.dispose();
    super.dispose();
  }
}
