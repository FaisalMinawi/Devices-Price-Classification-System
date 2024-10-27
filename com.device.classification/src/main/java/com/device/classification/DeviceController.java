package com.device.classification;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.transaction.annotation.Transactional;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;

@RestController
@RequestMapping("/api/devices")
public class DeviceController {

    @Autowired
    private DeviceRepository deviceRepository;

    @GetMapping
    public List<Device> getAllDevices() {
        return deviceRepository.findAll();
    }

    @GetMapping("/{id}")
    public ResponseEntity<Device> getDeviceById(@PathVariable Long id) {
        return deviceRepository.findById(id).map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public Device createDevice(@RequestBody Device device) {
        return deviceRepository.save(device);
    }

    @PostMapping("/predict/{deviceId}")
    @Transactional
    public ResponseEntity<Device> predictPrice(@PathVariable Long deviceId) {
        Device device = deviceRepository.findById(deviceId)
                .orElseThrow(() -> new ResourceNotFoundException("Device not found"));

        // Call Python API to predict price
        RestTemplate restTemplate = new RestTemplate();
        String pythonApiUrl = "http://localhost:8000/predict_price/";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Map the device to JSON without the ID and predicted price range fields
        HttpEntity<String> request = new HttpEntity<>(String.format(
                "{\"batteryPower\":%d, \"blue\":%b, \"clockSpeed\":%f, \"dualSim\":%b, \"fc\":%d, \"fourG\":%b, \"intMemory\":%d, \"mDep\":%f, \"mobileWt\":%d, \"nCores\":%d, \"pc\":%d, \"pxHeight\":%d, \"pxWidth\":%d, \"ram\":%d, \"scH\":%d, \"scW\":%d, \"talkTime\":%d, \"threeG\":%b, \"touchScreen\":%b, \"wifi\":%b}",
                device.getBatteryPower(),
                device.isBlue(),
                device.getClockSpeed(),
                device.isDualSim(),
                device.getFc(),
                device.isFourG(),
                device.getIntMemory(),
                device.getMDep(),
                device.getMobileWt(),
                device.getNCores(),
                device.getPc(),
                device.getPxHeight(),
                device.getPxWidth(),
                device.getRam(),
                device.getScH(),
                device.getScW(),
                device.getTalkTime(),
                device.isThreeG(),
                device.isTouchScreen(),
                device.isWifi()
        ), headers);

        ResponseEntity<String> response = restTemplate.postForEntity(pythonApiUrl, request, String.class);

        // Parse response and update the device with the predicted price
        try {
            // Parse JSON response
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode jsonResponse = objectMapper.readTree(response.getBody());

            // Extract predicted price range
            int predictedPriceRange = jsonResponse.get("predicted_price_range").asInt();
            device.setPredictedPriceRange(predictedPriceRange);
        } catch (Exception e) {
            throw new RuntimeException("Invalid response from prediction API: " + response.getBody(), e);
        }

        // Save updated device
        deviceRepository.save(device);
        return ResponseEntity.ok(device);
    }

    // Add the batch prediction method here
    @PostMapping("/predict/batch_all")
    public ResponseEntity<List<Device>> predictPriceForBatch() {
        List<Device> devices = deviceRepository.findAll().subList(0, Math.min(10, deviceRepository.findAll().size()));
        devices.forEach(device -> {
            try {
                predictPrice(device.getId());
            } catch (Exception e) {
                // Log error for specific device but continue with others
                System.err.println("Error predicting price for device ID " + device.getId() + ": " + e.getMessage());
            }
        });
        return ResponseEntity.ok(devices);
    }
}
