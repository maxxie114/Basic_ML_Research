#!/bin/bash

curl -X POST http://localhost:11434/api/chat \
	-H "Content-Type: application/json" \
	-d '{
             "model": "deepseek-r1:70b",
	     "options": {},
	     "messages": [{"role": "user", "content": "please guide me step by step how to go about doing leetcode 200"}]
	    }'


