#!/bin/bash

curl -X POST http://localhost:11434/api/chat \
	-H "Content-Type: application/json" \
	-d '{
             "model": "deepseek-r1:671b",
	     "options": {},
	     "messages": [{"role": "user", "content": "write me a basic helloworld in java"}]
	    }'


