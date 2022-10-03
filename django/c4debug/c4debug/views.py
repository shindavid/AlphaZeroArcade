import django.core.handlers.wsgi
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def replay(request: django.core.handlers.wsgi.WSGIRequest):
    debug_file = request.FILES['debug_file']
    n = 5
    text = []
    for line in debug_file:
        line = line.decode('utf-8')
        text.append(line)
        n -= 1
        if n == 0:
            break
    return HttpResponse(f'<html><body>Request: {"".join(text)}</body></html>')
