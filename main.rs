use actix_web::{web, App, HttpServer, HttpResponse};
use log::info;
use serde_json;

// 正确的自定义错误类型定义 - 使用结构体而不是类型别名
#[derive(Debug)]
struct InferenceError {
    message: String,
}

// 为自定义错误类型实现构造函数
impl InferenceError {
    fn new(msg: &str) -> Self {
        InferenceError {
            message: msg.to_string(),
        }
    }
}

// 实现 Display trait
impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

// 实现 Error trait
impl std::error::Error for InferenceError {}

// 现在可以为自定义类型实现 ResponseError（不再违反孤儿规则）
impl actix_web::ResponseError for InferenceError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::InternalServerError().json(serde_json::json!({
            "success": false,
            "error": self.message
        }))
    }
}

// 全局应用状态
struct AppState {
    // 可以在这里添加共享状态
}

// 图像预处理函数（模拟）
fn preprocess_image(_image_path: &str) -> Result<Vec<f32>, InferenceError> {
    info!("Preprocessing image: {}", _image_path);
    
    // 模拟预处理过程
    let mock_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    Ok(mock_data)
}

// 推理函数（模拟）
fn run_inference(_input_data: Vec<f32>) -> Result<serde_json::Value, InferenceError> {
    info!("Running inference on input data");
    
    // 模拟推理过程
    let mock_predictions = vec![0.05, 0.15, 0.60, 0.20];
    
    let result = serde_json::json!({
        "success": true,
        "predictions": mock_predictions,
        "class_count": mock_predictions.len(),
        "defect_probability": mock_predictions[2],
        "has_defect": mock_predictions[2] > 0.5
    });
    
    Ok(result)
}

// 预测接口处理函数
async fn predict(
    _data: web::Data<AppState>,
    image_path: web::Path<String>,
) -> Result<HttpResponse, InferenceError> {
    info!("Received prediction request for image: {}", image_path);
    
    let input_data = preprocess_image(&image_path)?;
    let inference_result = run_inference(input_data)?;
    
    Ok(HttpResponse::Ok().json(inference_result))
}

// 健康检查接口（简化版本，移除 chrono 依赖）
async fn health_check() -> HttpResponse {
    info!("Health check requested");
    
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "welding_defect_detection",
        "version": "1.0.0"
    }))
}

// 服务信息接口
async fn info() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "name": "Welding Defect Detection API",
        "version": "1.0.0",
        "description": "REST API for welding defect detection using machine learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/{image_path}",
            "info": "/info"
        }
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 初始化日志
    env_logger::init();
    
    info!("Starting Welding Defect Detection Service...");
    
    // 创建应用状态
    let app_state = web::Data::new(AppState {});
    
    let host = "127.0.0.1";
    let port = 8080;
    
    info!("Server will run on: {}:{}", host, port);
    info!("Available endpoints:");
    info!("  GET  /health - Health check");
    info!("  GET  /info - Service information"); 
    info!("  GET  /predict/{{image_path}} - Predict welding defects");
    
    // 启动 HTTP 服务器
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/health", web::get().to(health_check))
            .route("/info", web::get().to(info))
            .route("/predict/{image_path:.*}", web::get().to(predict))
    })
    .bind((host, port))?
    .run()
    .await
}
