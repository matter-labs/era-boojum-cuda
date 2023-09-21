use criterion::measurement::{Measurement, ValueFormatter};
use criterion::Throughput;

use cudart::event::{elapsed_time, CudaEvent};
use cudart::stream::CudaStream;

// Allows criterion benchmarks to use CUDA event-based timings.
// Based on src/lib.rs from https://github.com/theHamsta/criterion-cuda.
pub struct CudaMeasurement<const INV_ELEMS: bool = false>;

// c.f. https://docs.rs/criterion/latest/criterion/measurement/trait.Measurement.html
impl<const INV_ELEMS: bool> Measurement for CudaMeasurement<INV_ELEMS> {
    type Intermediate = (CudaEvent, CudaEvent);
    type Value = f32;

    fn start(&self) -> Self::Intermediate {
        let stream = CudaStream::default();
        let start_event = CudaEvent::create().expect("Failed to create event");
        let end_event = CudaEvent::create().expect("Failed to create event");
        start_event
            .record(&stream)
            .expect("Could not record CUDA event");
        (start_event, end_event)
    }

    fn end(&self, events: Self::Intermediate) -> Self::Value {
        let (start_event, end_event) = events;
        let stream = CudaStream::default();
        end_event
            .record(&stream)
            .expect("Could not record CUDA event");
        stream.synchronize().expect("Failed to synchronize");
        elapsed_time(&start_event, &end_event).expect("Failed to measure time")
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0f32
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CudaEventElapsedTimeFormatter::<INV_ELEMS>
    }
}

// based on https://github.com/bheisler/criterion.rs/blob/master/src/measurement.rs
struct CudaEventElapsedTimeFormatter<const INV_ELEMS: bool = false>;

impl<const INV_ELEMS: bool> CudaEventElapsedTimeFormatter<INV_ELEMS> {
    fn bytes_per_second(&self, bytes: f64, typical: f64, values: &mut [f64]) -> &'static str {
        let bytes_per_second = bytes * (1e3 / typical);
        const K: f64 = 1024.0;
        let (denominator, unit) = if bytes_per_second < K {
            (1.0, "  B/s")
        } else if bytes_per_second < K * K {
            (K, "KiB/s")
        } else if bytes_per_second < K * K * K {
            (K * K, "MiB/s")
        } else {
            (K * K * K, "GiB/s")
        };

        for val in values {
            let bytes_per_second = bytes * (1e3 / *val);
            *val = bytes_per_second / denominator;
        }

        unit
    }

    fn bytes_per_second_decimal(
        &self,
        bytes: f64,
        typical: f64,
        values: &mut [f64],
    ) -> &'static str {
        let bytes_per_second = bytes * (1e3 / typical);
        const K: f64 = 1000.0;
        let (denominator, unit) = if bytes_per_second < K {
            (1.0, "  B/s")
        } else if bytes_per_second < K * K {
            (K, "KB/s")
        } else if bytes_per_second < K * K * K {
            (K * K, "MB/s")
        } else {
            (K * K * K, "GB/s")
        };

        for val in values {
            let bytes_per_second = bytes * (1e3 / *val);
            *val = bytes_per_second / denominator;
        }

        unit
    }

    fn elements_per_second(&self, elems: f64, typical: f64, values: &mut [f64]) -> &'static str {
        let elems_per_second = elems * (1e3 / typical);
        const K: f64 = 1000.0;
        let (denominator, unit) = if elems_per_second < K {
            (1.0, " elem/s")
        } else if elems_per_second < K * K {
            (K, "Kelem/s")
        } else if elems_per_second < K * K * K {
            (K * K, "Melem/s")
        } else {
            (K * K * K, "Gelem/s")
        };

        for val in values {
            let elems_per_second = elems * (1e3 / *val);
            *val = elems_per_second / denominator;
        }

        unit
    }

    fn second_per_element(&self, elems: f64, typical: f64, values: &mut [f64]) -> &'static str {
        let ms_per_elem = typical / elems;
        let (factor, unit) = if ms_per_elem < 1e-6 {
            (1e9, "ps/elem")
        } else if ms_per_elem < 1e-3 {
            (1e6, "ns/elem")
        } else if ms_per_elem < 1e0 {
            (1e3, "µs/elem")
        } else if ms_per_elem < 1e3 {
            (1e0, "ms/elem")
        } else {
            (1e-3, "s/elem")
        };

        for val in values {
            *val *= factor / elems;
        }

        unit
    }
}

impl<const INV_ELEMS: bool> ValueFormatter for CudaEventElapsedTimeFormatter<INV_ELEMS> {
    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        let (factor, unit) = if typical_value < 1e-6 {
            (1e9, "ps")
        } else if typical_value < 1e-3 {
            (1e6, "ns")
        } else if typical_value < 1e0 {
            (1e3, "µs")
        } else if typical_value < 1e3 {
            (1e0, "ms")
        } else {
            (1e-3, "s")
        };

        for val in values {
            *val *= factor;
        }

        unit
    }

    fn scale_throughputs(
        &self,
        typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match *throughput {
            Throughput::Bytes(bytes) => self.bytes_per_second(bytes as f64, typical_value, values),
            Throughput::BytesDecimal(bytes) => {
                self.bytes_per_second_decimal(bytes as f64, typical_value, values)
            }
            Throughput::Elements(elems) => {
                if INV_ELEMS {
                    self.second_per_element(elems as f64, typical_value, values)
                } else {
                    self.elements_per_second(elems as f64, typical_value, values)
                }
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        // no scaling is needed
        "ms"
    }
}
