import 'package:flutter/material.dart';
import 'api_service.dart';

void main() {
  runApp(const CropYieldApp());
}

class CropYieldApp extends StatelessWidget {
  const CropYieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Yield Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2E7D32)),
        useMaterial3: true,
        fontFamily: 'serif',
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final areaController = TextEditingController(text: 'Rwanda');
  final itemController = TextEditingController(text: 'Maize');
  final yearController = TextEditingController(text: '2010');
  final rainfallController = TextEditingController(text: '1200');
  final pesticidesController = TextEditingController(text: '95');
  final tempController = TextEditingController(text: '19.5');

  String resultText = '';
  bool isError = false;
  bool isLoading = false;

  @override
  void dispose() {
    areaController.dispose();
    itemController.dispose();
    yearController.dispose();
    rainfallController.dispose();
    pesticidesController.dispose();
    tempController.dispose();
    super.dispose();
  }

  Future<void> handlePredict() async {
    setState(() {
      isLoading = true;
      resultText = '';
      isError = false;
    });

    // Check nothing is blank before bothering the server
    final fields = {
      'Country': areaController.text,
      'Crop': itemController.text,
      'Year': yearController.text,
      'Rainfall': rainfallController.text,
      'Pesticides': pesticidesController.text,
      'Temperature': tempController.text,
    };

    final blank = fields.entries.where((e) => e.value.trim().isEmpty).toList();
    if (blank.isNotEmpty) {
      setState(() {
        isLoading = false;
        isError = true;
        resultText = 'Please fill in: ${blank.map((e) => e.key).join(', ')}';
      });
      return;
    }

    // Make sure the numeric fields really are numbers
    final year = int.tryParse(yearController.text.trim());
    final rainfall = double.tryParse(rainfallController.text.trim());
    final pesticides = double.tryParse(pesticidesController.text.trim());
    final temp = double.tryParse(tempController.text.trim());

    if (year == null || rainfall == null || pesticides == null || temp == null) {
      setState(() {
        isLoading = false;
        isError = true;
        resultText = 'Year, rainfall, pesticides and temperature must be numbers.';
      });
      return;
    }

    try {
      final prediction = await ApiService.predictYield(
        area: areaController.text.trim(),
        item: itemController.text.trim(),
        year: year,
        rainfall: rainfall,
        pesticides: pesticides,
        avgTemp: temp,
      );

      setState(() {
        isLoading = false;
        isError = false;
        resultText = '${prediction.toStringAsFixed(0)} hg/ha\n'
            '(${(prediction / 10000).toStringAsFixed(2)} tonnes per hectare)';
      });
    } catch (e) {
      setState(() {
        isLoading = false;
        isError = true;
        resultText = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Yield Predictor'),
        backgroundColor: const Color(0xFF2E7D32),
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Predict crop yield for African countries',
              style: TextStyle(fontSize: 15, color: Colors.black54),
            ),
            const SizedBox(height: 20),

            buildField(areaController, 'Country', 'e.g. Rwanda'),
            buildField(itemController, 'Crop', 'e.g. Maize'),
            buildField(yearController, 'Year', '1990 - 2030', numeric: true),
            buildField(rainfallController, 'Average rainfall (mm/year)',
                '0 - 3000', numeric: true),
            buildField(pesticidesController, 'Pesticides (tonnes)', '0 - 30000',
                numeric: true),
            buildField(tempController, 'Average temperature (°C)', '5 - 40',
                numeric: true),

            const SizedBox(height: 12),
            SizedBox(
              height: 50,
              child: ElevatedButton(
                onPressed: isLoading ? null : handlePredict,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF2E7D32),
                  foregroundColor: Colors.white,
                ),
                child: isLoading
                    ? const SizedBox(
                  height: 22,
                  width: 22,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white,
                  ),
                )
                    : const Text('Predict', style: TextStyle(fontSize: 17)),
              ),
            ),

            const SizedBox(height: 24),
            buildResultBox(),
          ],
        ),
      ),
    );
  }

  Widget buildField(
      TextEditingController controller,
      String label,
      String hint, {
        bool numeric = false,
      }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextField(
        controller: controller,
        keyboardType:
        numeric ? const TextInputType.numberWithOptions(decimal: true) : TextInputType.text,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: const OutlineInputBorder(),
          isDense: true,
          contentPadding:
          const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        ),
      ),
    );
  }

  Widget buildResultBox() {
    if (resultText.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.grey.shade300),
        ),
        child: const Text(
          'Enter values above and tap Predict',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.black54),
        ),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isError ? const Color(0xFFFDECEA) : const Color(0xFFE8F5E9),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isError ? Colors.red.shade200 : Colors.green.shade200,
        ),
      ),
      child: Column(
        children: [
          Text(
            isError ? 'Error' : 'Predicted Yield',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: isError ? Colors.red.shade900 : Colors.green.shade900,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            resultText,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: isError ? 14 : 20,
              fontWeight: isError ? FontWeight.normal : FontWeight.bold,
              color: isError ? Colors.red.shade900 : Colors.green.shade900,
            ),
          ),
        ],
      ),
    );
  }
}