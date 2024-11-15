import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart'
    as mlkit;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:platform/platform.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'package:google_ml_vision/google_ml_vision.dart';

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

  late mlkit.FaceDetector _faceDetector;
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
  late Platform platform;
  // 디바이스 방향에 따른 회전 매핑
  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); // 앱 생명주기 관찰자 등록
    _initializeCamera(); // 카메라 초기화
    _loadModel(); // AI 모델 로드
    _initializeBuffers(); // 입력 버퍼 초기화
    _initializeFaceDetector();
  }

  void _initializeFaceDetector() {
    final options = mlkit.FaceDetectorOptions(
      enableLandmarks: true,
      performanceMode: mlkit.FaceDetectorMode.fast,
    );
    _faceDetector = mlkit.FaceDetector(options: options);
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

        imageFormatGroup: platform.isAndroid
            ? ImageFormatGroup.nv21 // Android용 이미지 포맷
            : ImageFormatGroup.bgra8888, // iOS용 이미지 포맷
      );
      await _controller.initialize();
      await _controller.startImageStream(_processCameraImage);

      setState(() {});
    } on CameraException catch (e) {
      print('카메라 초기화 오류: $e');
    }
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    final camera = widget.cameras[1];
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (platform.isAndroid) {
      var rotationCompensation =
          _orientations[_controller.value.deviceOrientation];
      if (rotationCompensation == null) return null;

      if (camera.lensDirection == CameraLensDirection.front) {
        // 전면 카메라
        rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      } else {
        // 후면 카메라
        rotationCompensation =
            (sensorOrientation - rotationCompensation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }
    if (rotation == null) return null;
    // 이미지 포맷 설정
    final format = platform.isAndroid
        ? InputImageFormat.nv21 // Android
        : InputImageFormat.bgra8888; // iOS

    // 이미지 평면이 하나만 있는지 확인
    if (image.planes.length != 1) return null;
    final plane = image.planes.first;

    // InputImage 생성
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
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
      final inputImage = _inputImageFromCameraImage(image);
      if (inputImage == null) return;

      // 얼굴 검출
      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isEmpty) return;

      // 나머지 처리 로직...
      final face = faces.first;
      final leftEye = face.landmarks[FaceLandmarkType.leftEye];
      final rightEye = face.landmarks[FaceLandmarkType.rightEye];

      if (leftEye == null || rightEye == null) return;

      // 눈 영역 추출 및 처리...
      final leftEyeImage = _extractEyeRegion(image, leftEye);
      final rightEyeImage = _extractEyeRegion(image, rightEye);

      final leftEyeResult = await _runInference(leftEyeImage);
      final rightEyeResult = await _runInference(rightEyeImage);

      final isEyesClosed = leftEyeResult[0] > 0.5 && rightEyeResult[0] > 0.5;

      if (mounted) {
        setState(() {
          if (isEyesClosed) {
            drowsyFrameCount++;
            if (drowsyFrameCount >= drowsyFrameThreshold) {
              _handleDrowsiness();
            }
          } else {
            drowsyFrameCount = max(0, drowsyFrameCount - 2);
          }
        });
      }
    } catch (e) {
      print('프레임 처리 오류: $e');
    }
  }

  List<List<List<double>>> _extractEyeRegion(
      CameraImage image, mlkit.FaceLandmark eye) {
    try {
      // 모델 입력 크기
      const int targetSize = 160;

      // 눈 영역 주변의 패딩 (픽셀)
      const int eyePadding = 10;

      // 눈 위치 계산 (카메라 이미지 크기에 맞게 스케일링)
      int centerX = (eye.position.x * image.width).toInt();
      int centerY = (eye.position.y * image.height).toInt();

      // 눈 영역의 바운딩 박스 계산
      int startX = max(0, centerX - targetSize ~/ 4 - eyePadding);
      int startY = max(0, centerY - targetSize ~/ 4 - eyePadding);
      int endX = min(image.width - 1, centerX + targetSize ~/ 4 + eyePadding);
      int endY = min(image.height - 1, centerY + targetSize ~/ 4 + eyePadding);

      // 결과 버퍼 초기화 (160x160x3)
      List<List<List<double>>> eyeBuffer = List.generate(
        targetSize,
        (_) => List.generate(
          targetSize,
          (_) => List.filled(3, 0.0),
          growable: false,
        ),
        growable: false,
      );

      // YUV420 이미지에서 RGB 추출 및 리사이징
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel!;

      // 실제 눈 영역의 크기
      final int eyeWidth = endX - startX;
      final int eyeHeight = endY - startY;

      // 스케일 팩터 계산
      final double scaleX = eyeWidth / targetSize;
      final double scaleY = eyeHeight / targetSize;

      // 타겟 이미지로 리샘플링
      for (int y = 0; y < targetSize; y++) {
        for (int x = 0; x < targetSize; x++) {
          // 원본 이미지에서의 위치 계산
          int sourceX = (startX + (x * scaleX)).toInt();
          int sourceY = (startY + (y * scaleY)).toInt();

          // YUV 값 추출
          final int uvIndex =
              uvPixelStride * (sourceX ~/ 2) + uvRowStride * (sourceY ~/ 2);
          final int index = sourceY * image.width + sourceX;

          // YUV to RGB 변환
          int yp = image.planes[0].bytes[index] & 0xFF;
          int up = image.planes[1].bytes[uvIndex] & 0xFF;
          int vp = image.planes[2].bytes[uvIndex] & 0xFF;

          // YUV -> RGB 변환 (BT.601 표준)
          int y1 = yp;
          int u = up - 128;
          int v = vp - 128;

          // RGB 계산 (최적화된 정수 연산)
          int r = y1 + ((1402 * v) >> 10);
          int g = y1 - ((344 * u + 714 * v) >> 10);
          int b = y1 + ((1772 * u) >> 10);

          // 값 범위 제한
          r = r.clamp(0, 255);
          g = g.clamp(0, 255);
          b = b.clamp(0, 255);

          // MobileNetV2 전처리 (-1 ~ 1 범위로 정규화)
          eyeBuffer[y][x][0] = (r / 127.5) - 1.0; // R
          eyeBuffer[y][x][1] = (g / 127.5) - 1.0; // G
          eyeBuffer[y][x][2] = (b / 127.5) - 1.0; // B
        }
      }

      // 디버깅을 위한 통계 출력

      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      double sum = 0;
      int count = 0;

      for (var row in eyeBuffer) {
        for (var pixel in row) {
          for (var value in pixel) {
            minVal = min(minVal, value);
            maxVal = max(maxVal, value);
            sum += value;
            count++;
          }
        }
      }

      print('Eye region statistics:');
      print('Min: $minVal, Max: $maxVal, Mean: ${sum / count}');

      return eyeBuffer;
    } catch (e) {
      print('Error in eye region extraction: $e');
      // 오류 발생 시 기본값 반환
      return List.generate(
        160,
        (_) => List.generate(
          160,
          (_) => List.filled(3, 0.0),
          growable: false,
        ),
        growable: false,
      );
    }
  }

// 스무딩을 위한 보조 함수
  double _interpolate(double value, double targetMin, double targetMax,
      double sourceMin, double sourceMax) {
    return targetMin +
        (value - sourceMin) * (targetMax - targetMin) / (sourceMax - sourceMin);
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

// 추론 결과를 보정하기 위한 이동 평균 필터 구현
class MovingAverageFilter {
  final int windowSize;
  final List<List<double>> buffer;
  int currentIndex = 0;
  bool isFull = false;

  MovingAverageFilter(this.windowSize)
      : buffer = List.generate(windowSize, (_) => [0.0, 0.0]);

  List<double> update(List<double> newValue) {
    buffer[currentIndex] = newValue;
    currentIndex = (currentIndex + 1) % windowSize;
    isFull = isFull || currentIndex == 0;

    if (!isFull) {
      // 버퍼가 채워지지 않은 경우 현재까지의 평균 계산
      List<double> sum = [0.0, 0.0];
      for (int i = 0; i < currentIndex; i++) {
        sum[0] += buffer[i][0];
        sum[1] += buffer[i][1];
      }
      return [sum[0] / currentIndex, sum[1] / currentIndex];
    } else {
      // 전체 윈도우의 평균 계산
      List<double> sum = [0.0, 0.0];
      for (var value in buffer) {
        sum[0] += value[0];
        sum[1] += value[1];
      }
      return [sum[0] / windowSize, sum[1] / windowSize];
    }
  }
}
