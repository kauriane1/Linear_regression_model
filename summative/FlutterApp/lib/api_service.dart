import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl =
      'https://linear-regression-model-26zx.onrender.com';

  /// Sends the six input values to the API and returns the predicted yield.
  /// Throws with a readable message if the API rejects the input.
  static Future<double> predictYield({
    required String area,
    required String item,
    required int year,
    required double rainfall,
    required double pesticides,
    required double avgTemp,
  }) async {
    final uri = Uri.parse('$baseUrl/predict');

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'Area': area,
        'Item': item,
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': avgTemp,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return (data['predicted_yield_hg_per_ha'] as num).toDouble();
    }

    // The API returns 422 with details when validation fails
    if (response.statusCode == 422) {
      throw Exception(_readableError(response.body));
    }

    throw Exception('Server error (${response.statusCode}). Please try again.');
  }

  /// The API returns two shapes of 422: a plain string for our own checks,
  /// and a list of field errors from Pydantic. Handle both.
  static String _readableError(String body) {
    try {
      final decoded = jsonDecode(body);
      final detail = decoded['detail'];

      if (detail is String) return detail;

      if (detail is List && detail.isNotEmpty) {
        final first = detail.first;
        final field = (first['loc'] as List).last;
        return '$field: ${first['msg']}';
      }
    } catch (_) {
      // fall through
    }
    return 'Invalid input. Please check your values.';
  }
}