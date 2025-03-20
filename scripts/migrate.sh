#!/bin/bash

# Run database migrations
python -m multiagent.app.db.db_cli migrate

# Exit with the status of the migrations
exit $?