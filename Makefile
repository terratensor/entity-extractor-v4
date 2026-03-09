.PHONY: clean test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "🧹 Кэш Python очищен"

test: clean
	-rm ./results_test/output.csv
	-rm batch_processor_v4.log
	-rm checkpoint_test.json
	./run_test.sh