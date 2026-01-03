from __future__ import annotations

from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
import redis as redis_sync
# from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
import redis.asyncio as redis
from fastapi.middleware.cors import CORSMiddleware

# from RAG_API_Clean.core.settings_store import APP_TITLE

# from RAG_API_Clean.core.settings_store import APP_TITLE, load_settings, load_env_from_settings
# from RAG_API_Clean.core.memory_store import log, set_redis_client
# from RAG_API_Clean.app.api import router, PROJECT_ROOT
from app.api import router, PROJECT_ROOT
from app.history_api import router as history_router
from core.memory_store import log, set_redis_client
from core.settings_store import APP_TITLE, load_settings, load_env_from_settings

# app = FastAPI(title=APP_TITLE)

# app = FastAPI()

app = FastAPI(
    # filter = True,
    title=APP_TITLE,
    version="0.1.0",
    swagger_ui_parameters={
        "filter": True,
        
    },
)

# app = FastAPI(title=APP_TITLE, docs_url="/redoc", redoc_url="/redoc", swagger_ui=None)  # Or docs_url="/docs" and redirect

# @app.get("/docs1", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title=app.title + " - Swagger UI",
#         swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
#         swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
#         swagger_ui_parameters={
#             "dom_id": "#swagger-ui",
#             "layout": "BaseLayout",
#             "deepLinking": True,
#             "filter": True  # Enables the built-in filter input (hides until you type)
#         }
#     )

# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return HTMLResponse("""
#     <!doctype html>
#     <html lang="en">
#     <head>
#         <meta charset="utf-8">
#         <title>Swagger UI with Search</title>
#         <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
#     </head>
#     <body>
#         <div id="swagger-ui"></div>
#         <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
#         <!-- Removed standalone preset script (not needed and causes issues in v5) -->

#         <script>
#         window.onload = function() {
#             const SearchPlugin = () => {
#                 return {
#                     afterLoad: function(system) {
#                         const observer = new MutationObserver(() => {
#                             const container = document.querySelector("#swagger-ui > section > div.swagger-ui > div:nth-child(2) > div.scheme-container");
#                             if (container && !document.getElementById('endpoint-search')) {
#                                 const searchBar = document.createElement("div");
#                                 searchBar.id = "search-container";
#                                 searchBar.style.margin = "20px 0";
#                                 searchBar.style.padding = "0 20px";

#                                 const searchInput = document.createElement("input");
#                                 searchInput.type = "text";
#                                 searchInput.id = "endpoint-search";
#                                 searchInput.placeholder = "Search by endpoint path or summary...";
#                                 searchInput.style.width = "100%";
#                                 searchInput.style.padding = "10px";
#                                 searchInput.style.boxSizing = "border-box";
#                                 searchInput.style.marginBottom = "10px";
#                                 searchInput.oninput = filterEndpoints;

#                                 searchBar.appendChild(searchInput);
#                                 container.after(searchBar);
#                                 observer.disconnect();
#                             }
#                         });
#                         observer.observe(document.body, { childList: true, subtree: true });
#                     }
#                 };
#             };

#             function filterEndpoints() {
#                 const input = document.getElementById('endpoint-search').value.toLowerCase();
#                 const operations = document.querySelectorAll('.opblock');

#                 operations.forEach((operation) => {
#                     const summary = operation.querySelector('.opblock-summary-description')?.textContent.toLowerCase() || '';
#                     const path = operation.querySelector('.opblock-summary-path')?.textContent.toLowerCase() || '';

#                     if (input === '' || summary.includes(input) || path.includes(input)) {
#                         operation.style.display = '';
#                     } else {
#                         operation.style.display = 'none';
#                     }
#                 });
#             }

#             const ui = SwaggerUIBundle({
#                 url: "/openapi.json",
#                 dom_id: '#swagger-ui',
#                 deepLinking: true,
#                 presets: [
#                     SwaggerUIBundle.presets.apis
#                     // No StandalonePreset needed
#                 ],
#                 plugins: [
#                     SearchPlugin
#                 ],
#                 layout: "BaseLayout"  // Default layout - works perfectly with CDN
#             });
#         };
#         </script>
#     </body>
#     </html>
#     """)

# Redis client (optional, used for RAG history + logs/results if you wire them)
@app.on_event("startup")
async def _startup():
    try:
        settings = load_settings(PROJECT_ROOT)
        redis_url = settings.get("redis_url") or "redis://localhost:6379/0"
        app.state.redis = redis.from_url(redis_url, decode_responses=True)
        app.state.redis = redis.from_url(redis_url, decode_responses=True)        # async (history)
        app.state.redis_sync = redis_sync.from_url(redis_url, decode_responses=True)  # sync (logs/results)        
        set_redis_client(app.state.redis, prefix="rag")
        # ping to validate connection (doesn't crash app if redis is down)
        try:
            await app.state.redis.ping()
            log(f"[Redis] Connected OK: {redis_url}")
        except Exception as e:
            log(f"[Redis] Unreachable: {e}")
    except Exception as e:
        app.state.redis = None
        set_redis_client(None)
        log(f"[Redis] Init error: {e}")


@app.on_event("shutdown")
async def _shutdown():
    r = getattr(app.state, "redis", None)
    if r is not None:
        try:
            await r.close()
        except Exception:
            pass

# Optional: CORS (useful for frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, tags=["settings"])
app.include_router(history_router, tags=["History"])

@app.on_event("startup")
def on_startup():
    s = load_settings(PROJECT_ROOT)
    load_env_from_settings(PROJECT_ROOT, s)
    log(f"API started. Settings loaded from {PROJECT_ROOT / 'rag_settings.json'}")
    log(f"Qdrant={s.get('qdrant_url')} prefix={s.get('collection_prefix')}")
    log(f"Providers: main={s.get('main_provider')} embedding={s.get('embedding_provider')}")
    log(f"Models: embedding={s.get('embedding_model')} main={s.get('main_model')}")
